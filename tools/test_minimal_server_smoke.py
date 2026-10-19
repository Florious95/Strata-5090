"""Skip-aware runtime smoke accounting must never turn skips/errors into passes."""
import subprocess
import unittest
from unittest import mock
from tools import minimal_server_smoke as smoke


class RuntimeSmokeAccounting(unittest.TestCase):
    def report(self, summary, exit_code=0):
        visible = {name: object() if name not in ('rich', 'textual') else None for name in
                   ('rich', 'textual', 'numpy', 'jinja2', 'requests')}
        result = subprocess.CompletedProcess([], exit_code, '', summary)
        with mock.patch.object(smoke.sys, 'prefix', '/isolated-venv'), \
                mock.patch.object(smoke.os, 'name', 'posix'), \
                mock.patch.object(smoke.importlib.util, 'find_spec', side_effect=visible.get), \
                mock.patch.object(smoke.subprocess, 'run', return_value=result):
            return smoke.smoke()

    def test_success_with_platform_skips_counts_only_executed_passes(self):
        report = self.report('Ran 39 tests in 10.0s\n\nOK (skipped=3)\n')
        self.assertTrue(report['passed'])
        self.assertEqual((report['tests_run'], report['tests_passed'], report['tests_skipped']), (39, 36, 3))

    def test_success_without_skips(self):
        self.assertTrue(self.report('Ran 39 tests in 10.0s\n\nOK\n')['passed'])

    def test_skips_do_not_satisfy_minimum_executed_cases(self):
        self.assertFalse(self.report('Ran 39 tests in 10.0s\n\nOK (skipped=8)\n')['passed'])

    def test_nonzero_exit_is_never_success(self):
        self.assertFalse(self.report('Ran 39 tests in 10.0s\n\nOK\n', 1)['passed'])
        self.assertFalse(self.report('Ran 39 tests in 10.0s\n\nFAILED (errors=1)\n', 1)['passed'])


if __name__ == '__main__':
    unittest.main()
