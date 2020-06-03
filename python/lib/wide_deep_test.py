from absl import logging
import tensorflow as tf
import numpy as np

import os
import sys
PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACKAGE_DIR)

from lib.read_conf import Config
from lib.dataset import input_fn
from lib.build_estimator import build_custom_estimator, build_estimator


TEST_CSV = os.path.join(os.path.dirname(PACKAGE_DIR), 'data/test/test2')
USED_FEATURE_KEY = Config().get_feature_name('used')


def _read_test_input(all_lines=False):
    if all_lines:
        return open(TEST_CSV).readlines()
    else:
        return open(TEST_CSV).readline()


TEST_INPUT_VALUES = _read_test_input()
TEST_INPUT_KEYS = Config().get_feature_name()
TEST_INPUT = dict(zip(TEST_INPUT_KEYS, TEST_INPUT_VALUES.strip().split("\t")[1:]))
for key in TEST_INPUT:
    TEST_INPUT[key] = TEST_INPUT[key].split(',')


class BaseTest(tf.test.TestCase):

    def setUp(self):
        # Create temporary CSV file
        self.temp_dir = self.get_temp_dir()
        self.input_csv = os.path.join(self.temp_dir, 'test.csv')
        with tf.io.gfile.GFile(self.input_csv, 'w') as temp_csv:
            for line in _read_test_input(True):
                temp_csv.write(line.strip() + "\n")

    def test_input_fn(self):
        tf.compat.v1.enable_eager_execution()
        features, labels = input_fn(self.input_csv, None, mode='eval', batch_size=1)
        # Compare the two features dictionaries.
        for KEY in USED_FEATURE_KEY:
            self.assertTrue(KEY in features)
            self.assertEqual(len(features[KEY][0]), len(TEST_INPUT[KEY]))

            feature_values = features[KEY][0].numpy()
            print(KEY, TEST_INPUT[KEY], feature_values)
            # Convert from bytes to string for Python 3.
            for i in range(len(TEST_INPUT[KEY])):
                feature_value = feature_values[i]
                if isinstance(feature_value, bytes):
                    feature_value = feature_value.decode()
                if isinstance(feature_value, np.int32):
                    feature_value = str(feature_value)
                if isinstance(feature_value, np.float32):
                    TEST_INPUT[KEY][i] = np.float32(TEST_INPUT[KEY][i])
                self.assertEqual(TEST_INPUT[KEY][i], feature_value)
        self.assertFalse(labels)

    def build_and_test_estimator(self, model_type):
        """Ensure that model trains and minimizes loss."""
        model = build_estimator(self.temp_dir, model_type)

        # Train for 1 step to initialize model and evaluate initial loss
        model.train(
            input_fn=lambda: input_fn(
                TEST_CSV, None, 'eval', batch_size=1),
            steps=1)
        initial_results = model.evaluate(
            input_fn=lambda: input_fn(
                TEST_CSV, None, 'eval', batch_size=1))

        # Train for 100 epochs at batch size 3 and evaluate final loss
        model.train(
            input_fn=lambda: input_fn(
                TEST_CSV, None, 'eval', batch_size=8))
        final_results = model.evaluate(
            input_fn=lambda: input_fn(
                TEST_CSV, None, 'eval', batch_size=1))

        logging.info('\n%s initial results: %s', model_type, initial_results)
        logging.info('\n%s final results: %s', model_type, final_results)

        # Ensure loss has decreased, while accuracy and both AUCs have increased.
        self.assertLess(final_results['loss'], initial_results['loss'])
        self.assertGreaterEqual(final_results['auc'], initial_results['auc'])
        self.assertGreaterEqual(final_results['auc_precision_recall'],
                           initial_results['auc_precision_recall'])
        self.assertGreaterEqual(final_results['accuracy'], initial_results['accuracy'])

    def test_wide_deep_estimator_training(self):
        self.build_and_test_estimator('wide_deep')


if __name__ == '__main__':
    logging.set_verbosity(logging.DEBUG)
    logging.set_verbosity(tf.logging.DEBUG)
    tf.test.main()
