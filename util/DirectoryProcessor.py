import os
import shutil
from typing import List, Tuple


class DirectoryProcessor:
    """A utility class for processing directories and files."""

    @staticmethod
    def show_structure(target_dir: str, padding: str = '') -> None:
        """
        Recursively displays the directory structure in a tree format.
        """
        try:
            dirs = sorted(os.listdir(target_dir))
            for dir_entry in dirs:
                file_path = os.path.join(target_dir, dir_entry)
                print(f"{padding}├── {dir_entry}")
                if os.path.isdir(file_path):
                    DirectoryProcessor.show_structure(file_path, padding + '│   ')
        except Exception as e:
            print(f"Error while showing structure: {e}")

    @staticmethod
    def show_structure_with_count(target_dir: str, padding: str = '') -> None:
        """
        Recursively displays the directory structure in a tree format
        with a count of files in each subdirectory.
        """
        try:
            dirs = sorted(os.listdir(target_dir))
            for dir_entry in dirs:
                file_path = os.path.join(target_dir, dir_entry)
                if os.path.isdir(file_path):
                    sub_files = [f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))]
                    num_files = len(sub_files)
                    print(f"{padding}├── {dir_entry} \t ({num_files} files)")
                    DirectoryProcessor.show_structure_with_count(file_path, padding + '│   ')
        except Exception as e:
            print(f"Error while showing structure with count: {e}")

    @staticmethod
    def get_dir(target_dir: str) -> List[str]:
        """
        Returns a list of all subdirectories in the specified directory.
        """
        try:
            return [
                os.path.join(target_dir, entry)
                for entry in os.listdir(target_dir)
                if os.path.isdir(os.path.join(target_dir, entry))
            ]
        except Exception as e:
            print(f"Error while getting directories: {e}")
            return []

    @staticmethod
    def get_all_files(target_dir: str, include_sub_dir: bool = False) -> List[str]:
        """
        Returns a list of all files in the specified directory.
        Optionally includes files in subdirectories.
        """
        try:
            if include_sub_dir:
                return [
                    os.path.join(root, file)
                    for root, _, files in os.walk(target_dir)
                    for file in files
                ]
            else:
                return [
                    os.path.join(target_dir, file)
                    for file in os.listdir(target_dir)
                    if os.path.isfile(os.path.join(target_dir, file))
                ]
        except Exception as e:
            print(f"Error while getting all files: {e}")
            return []

    @staticmethod
    def get_only_files(target_dir: str, extensions: List[str], include_sub_dir: bool = False) -> List[str]:
        """
        Returns a list of files with the specified extensions.
        Optionally includes files in subdirectories.
        """
        try:
            if include_sub_dir:
                return [
                    os.path.join(root, file)
                    for root, _, files in os.walk(target_dir)
                    for file in files
                    if any(file.endswith(ext) for ext in extensions)
                ]
            else:
                return [
                    os.path.join(target_dir, file)
                    for file in os.listdir(target_dir)
                    if os.path.isfile(os.path.join(target_dir, file)) and any(file.endswith(ext) for ext in extensions)
                ]
        except Exception as e:
            print(f"Error while getting files with extensions: {e}")
            return []

    @staticmethod
    def decompose_path(path: str) -> Tuple[str, str, str]:
        """
        Decomposes a file path into directory, filename, and extension.
        """
        directory, filename = os.path.split(path)
        filename, extension = os.path.splitext(filename)
        return directory, filename, extension

    @staticmethod
    def path_up(path: str) -> Tuple[str, str]:
        """
        Returns the parent directory and the current folder or file name.
        """
        parent_path = os.path.dirname(path)
        current_name = os.path.basename(path)
        return parent_path, current_name

    @staticmethod
    def create_dir(directory_path: str) -> None:
        """
        Creates a directory if it does not already exist.
        """
        try:
            os.makedirs(directory_path, exist_ok=True)
        except Exception as e:
            print(f"Error while creating directory: {e}")

    @staticmethod
    def move_file(source: str, destination: str) -> None:
        """
        Moves a file from source to destination.
        Creates destination directory if it does not exist.
        """
        try:
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            shutil.move(source, destination)
        except Exception as e:
            print(f"Error while moving file: {e}")

    @staticmethod
    def copy_file(source: str, destination: str) -> None:
        """
        Copies a file from source to destination.
        Creates destination directory if it does not exist.
        """
        try:
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            shutil.copyfile(source, destination)
        except Exception as e:
            print(f"Error while copying file: {e}")

    @staticmethod
    def rename_file(old_name: str, new_name: str) -> None:
        """
        Renames a file from old_name to new_name.
        """
        try:
            os.rename(old_name, new_name)
        except Exception as e:
            print(f"Error while renaming file: {e}")
