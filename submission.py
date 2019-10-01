import os
from datetime import datetime
import csv

class SubmissionWriter:

    """ Class for collecting results and exporting submission. """

    def __init__(self):
        self.test_results = []
        self.real_test_results = []
        return

    def _append(self, filename, q, r, real):
        if real:
            self.real_test_results.append({'filename': filename, 'q': list(q), 'r': list(r)})
        else:
            self.test_results.append({'filename': filename, 'q': list(q), 'r': list(r)})
        return

    def append_test(self, filename, q, r):

        """ Append pose estimation for test image to submission. """

        self._append(filename, q, r, real=False)
        return

    def append_real_test(self, filename, q, r):

        """ Append pose estimation for real image to submission. """

        self._append(filename, q, r, real=True)
        return

    def export(self, out_dir='', suffix=None):

        """ Exporting submission json file containing the collected pose estimates. """

        sorted_test = sorted(self.test_results, key=lambda k: k['filename'])
        sorted_real_test = sorted(self.real_test_results, key=lambda k: k['filename'])
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        if suffix is None:
            suffix = timestamp
        submission_path = os.path.join(out_dir, 'submission_{}.csv'.format(suffix))
        with open(submission_path, 'w') as f:
            csv_writer = csv.writer(f, lineterminator='\n')
            for result in (sorted_test + sorted_real_test):
                csv_writer.writerow([result['filename'], *(result['q'] + result['r'])])

        print('Submission saved to {}.'.format(submission_path))
        return
