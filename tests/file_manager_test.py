import unittest

from src.core import file_manager


class FileManagerTestCase(unittest.TestCase):
    def test_not_empty_root_dir(self):
        self.assertTrue(bool(file_manager.get_project_dir()))

    def test_start_root_dir_(self):
        root_character = '/'

        self.assertTrue(file_manager.get_project_dir().startswith(root_character))

    def test_end_root_dir_(self):
        project_name = 'intent_classifier'

        self.assertTrue(file_manager.get_project_dir().endswith(project_name))

    def test_generate_filename_from_root_dir(self):
        self.assertEqual(file_manager.filename_from_project_dir('foo.py'), f'{file_manager.get_project_dir()}/foo.py')

    def test_generate_filename_from_data_dir(self):
        self.assertEqual(file_manager.filename_from_data_dir('foo.py'), f'{file_manager.get_project_dir()}/data/foo.py')


if __name__ == '__main__':
    unittest.main()
