#!/usr/bin/env python

"""
Copyright 2018 wfwiggins203 (https://github.com/wfwiggins203/)
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
import os
import sys

import keras
import tensorflow as tf

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin  # noqa: F401
    __package__ = "keras_retinanet.bin"

# Change these to absolute imports if you copy this script outside the keras_retinanet package.
from .. import models
from ..preprocessing.csv_generator_dcm import CSVGenerator
from ..utils.config import read_config_file, parse_anchor_parameters
from ..utils.eval_dcm import evaluate, _score_detections
from ..utils.keras_version import check_keras_version


def get_session():
    """ Construct a modified tf session.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def create_generator(args):
    """ Create generators for evaluation.
    """
    if args.dataset_type == 'csv':
        validation_generator = CSVGenerator(
            args.annotations,
            args.classes,
            image_min_side=args.image_min_side,
            image_max_side=args.image_max_side,
            config=args.config
        )
    else:
        raise ValueError('Invalid data type received: {}'.format(args.dataset_type))

    return validation_generator


def parse_args(args):
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Evaluation script for a RetinaNet network.')
    subparsers = parser.add_subparsers(help='Arguments for specific dataset types.', dest='dataset_type')
    subparsers.required = True

    csv_parser = subparsers.add_parser('csv')
    csv_parser.add_argument('annotations', help='Path to CSV file containing annotations for evaluation.')
    csv_parser.add_argument('classes', help='Path to a CSV file containing class label mapping.')

    parser.add_argument('model',              help='Path to RetinaNet model.')
    parser.add_argument('--convert-model',    help='Convert the model to an inference model (ie. the input is a training model).', action='store_true')
    parser.add_argument('--backbone',         help='The backbone of the model.', default='resnet50')
    parser.add_argument('--gpu',              help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--score-threshold',  help='Threshold on score to filter detections with (defaults to 0.20).', default=0.20, type=float)
    parser.add_argument('--max-detections',   help='Max Detections per image (defaults to 100).', default=100, type=int)
    parser.add_argument('--save-path',        help='Path for saving images with detections (doesn\'t work for COCO).')
    parser.add_argument('--image-min-side',   help='Rescale the image so the smallest side is min_side.', type=int, default=512)
    parser.add_argument('--image-max-side',   help='Rescale the image if the largest side is larger than max_side.', type=int, default=512)
    parser.add_argument('--config',           help='Path to a configuration parameters .ini file.')
    parser.add_argument('--weighted-average', help='Compute the mAP using the weighted average of precisions among classes.', action='store_true')

    return parser.parse_args(args)


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # make sure keras is the minimum required version
    check_keras_version()

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    # make save path if it doesn't exist
    if args.save_path is not None and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # optionally load config parameters
    if args.config:
        args.config = read_config_file(args.config)

    # create the generator
    print('Creating generator... This may take a minute.')
    generator = create_generator(args)

    # optionally load anchor parameters
    anchor_params = None
    if args.config and 'anchor_parameters' in args.config:
        anchor_params = parse_anchor_parameters(args.config)

    # load the model
    print('Loading model...')
    model = models.load_model(args.model, backbone_name=args.backbone, convert=args.convert_model, anchor_params=anchor_params)

    # print model summary
    # print(model.summary())

    # start evaluation
    precisions, all_detections, all_annotations = evaluate(
        generator,
        model,
        score_threshold=args.score_threshold,
        max_detections=args.max_detections,
        save_path=args.save_path
    )

    # loop over scores thresholds ranging from 0.20 to 0.50 at 0.025 intervals
    score_thresholds = [args.score_threshold + 0.025 * i for i in range(13)]

    for score in score_thresholds:
        if score > args.score_threshold:
            precisions = _score_detections(generator.size(), all_detections, all_annotations, score_threshold=score)
        print("\n")
        # print evaluation
        total_instances = []
        avg_prec = []
        for thresh, (precision, num_annotations) in precisions.items():
            print('{:.0f} instances of pneumonia'.format(num_annotations),
                'with precision: {:.4f} at IOU threshold {:.2f}'.format(precision, thresh))
            total_instances.append(num_annotations)
            avg_prec.append(precision)

        if sum(total_instances) == 0:
            print('No test instances found.')
            return

        if args.weighted_average:
            print('AP: {:.4f} at score threshold {:.2f}'.format(sum([a * b for a, b in zip(total_instances, avg_prec)]) / sum(total_instances), score))
        else:
            print('AP: {:.4f} at score threshold {:.2f}'.format(sum(avg_prec) / sum(x > 0 for x in total_instances), score))


if __name__ == '__main__':
    main()
