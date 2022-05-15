import unittest
import pandas as pd

from src.core import file_manager as fm


class FileManagerTestCase(unittest.TestCase):
    def test_get_project_dir(self):
        self.assertTrue(bool(fm.get_project_dir()))

        self.assertTrue(fm.get_project_dir().startswith('/'))

        self.assertTrue(fm.get_project_dir().endswith('intent_classifier'))

    def test_filename_from_project_dir(self):
        self.assertEqual(fm.filename_from_project_dir('foo.py'), f'{fm.get_project_dir()}/foo.py')

    def test_filename_from_data_dir(self):
        self.assertEqual(fm.filename_from_data_dir('foo.py'), f'{fm.get_project_dir()}/data/foo.py')

    def test_read_read_json_of_dir(self):
        directory_with_one_file = fm.filename_from_project_dir('tests/data/directory_with_one_file')
        df = fm.read_json_of_dir(directory_with_one_file)
        df_expected = pd.read_json(f'{directory_with_one_file}/messages_01.json')

        self.assertEqual(df.equals(df_expected), True)

        directory_with_two_files = fm.filename_from_project_dir('tests/data/directory_with_two_files')
        df = fm.read_json_of_dir(directory_with_two_files)
        df_expected = pd.concat([
            pd.read_json(f'{directory_with_two_files}/messages_02.json'),
            pd.read_json(f'{directory_with_two_files}/messages_03.json'),
        ])

        self.assertEqual(df.equals(df_expected), True)


if __name__ == '__main__':
    unittest.main()
