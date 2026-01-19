import os
import re
import shutil
from typing import List, Tuple

class DirectoryProcessor:
    """A utility class for processing directories and files."""

    @staticmethod
    def remove_file(path: str) -> bool:
        try:
            os.remove(path)
            return True
        except OSError as e:
            print(f"Error: {path} : {e.strerror}")
            return False
    
    @staticmethod
    def remove_dir(path: str) -> bool:
        """Removes a directory. Use shutil.rmtree for recursive delete."""
        try:
            shutil.rmtree(path)
            return True
        except OSError as e:
            print(f"Error: {path} : {e.strerror}")
            return False
    
    @staticmethod
    def remove_empty_dirs(target_dir: str) -> None:
        """Recursively removes empty subdirectories."""
        for root, dirs, files in os.walk(target_dir, topdown=False):
            for name in dirs:
                try:
                    os.rmdir(os.path.join(root, name))
                except OSError:
                    # Directory not empty
                    pass
        
    @staticmethod
    def show_structure(
        target_dir: str, 
        padding: str = '', 
        depth: int = 0, 
        max_depth: int = 3, 
        max_dirs: int = 10, 
        max_files: int = 10, 
        show_counts: bool = False,   # <--- New toggle
        exclude_dirs: List[str] = None
    ) -> None:
        """
        Unified directory tree viewer.
        
        Args:
            max_files: Set to 0 to hide filenames entirely (folder-only view).
            show_counts: If True, calculates and displays file counts next to folder names.
        """
        if exclude_dirs is None:
            exclude_dirs = ['.git', '__pycache__', 'node_modules', 'venv', '.idea', '__MACOSX']

        if depth > max_depth:
            return

        try:
            # 1. Gather and Sort Items
            try:
                items = os.listdir(target_dir)
            except PermissionError:
                print(f"{padding}├── 🔒 [Access Denied]")
                return

            dirs = []
            files = []
            for item in items:
                if item in exclude_dirs: continue
                # We use full path to check isdir/isfile safely
                full_path = os.path.join(target_dir, item)
                if os.path.isdir(full_path):
                    dirs.append(item)
                else:
                    files.append(item)
            
            dirs.sort()
            files.sort()

            # 2. Process Directories
            displayed_dirs = dirs[:max_dirs]
            remaining_dirs = len(dirs) - max_dirs

            for d in displayed_dirs:
                # OPTIONAL: Peek ahead to count files in the subdirectory
                count_info = ""
                if show_counts:
                    try:
                        sub_path = os.path.join(target_dir, d)
                        # Fast list comprehension to count files only
                        n_files = len([name for name in os.listdir(sub_path) 
                                     if os.path.isfile(os.path.join(sub_path, name))])
                        count_info = f" ({n_files} files)"
                    except (PermissionError, OSError):
                        count_info = " (?)"

                print(f"{padding}├── 📁 {d}{count_info}")

                # Recurse
                DirectoryProcessor.show_structure(
                    os.path.join(target_dir, d), 
                    padding + '│   ', 
                    depth + 1, 
                    max_depth, 
                    max_dirs, 
                    max_files, 
                    show_counts,
                    exclude_dirs
                )
            
            if remaining_dirs > 0:
                print(f"{padding}├── ... {remaining_dirs} more folders")

            # 3. Process Files
            # If max_files is 0, we skip listing them, but we can show a summary if files exist.
            if max_files == 0:
                if len(files) > 0 and depth == 0:
                    # Print a single summary line for the hidden files
                    print(f"{padding}└── 📄 ({len(files)} files)")
            else:
                displayed_files = files[:max_files]
                remaining_files = len(files) - max_files

                for f in displayed_files:
                    print(f"{padding}├── 📄 {f}")
                
                if remaining_files > 0:
                    print(f"{padding}└── ... {remaining_files} more files")

        except Exception as e:
            print(f"Error processing {target_dir}: {e}")

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
    def get_files_by_pattern(target_dir: str, pattern: str) -> List[str]:
        """Returns files matching a regex pattern."""
        res = []
        try:
            regex = re.compile(pattern)
            for file in os.listdir(target_dir):
                if os.path.isfile(os.path.join(target_dir, file)) and regex.match(file):
                    res.append(os.path.join(target_dir, file))
        except Exception as e:
            print(f"Error: {e}")
        return res

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
    def move_file(source: str, destination: str, overwrite: bool = False) -> None:
        """
        Moves a file from source to destination.
        Creates destination directory if it does not exist.
        """
        try:
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            if os.path.exists(destination) and not overwrite:
                print(f"Skipped: '{destination}' already exists (overwrite=False).")
                return
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

    @staticmethod
    def find_match_name(target_dir: str) -> None:
        """
        Moves files with the same name but different extensions in the target directory 
        to a subfolder named "MATCHED".
        """
        all_path = DirectoryProcessor.get_all_files(target_dir)
        filename_list = [DirectoryProcessor.decompose_path(path)[1] for path in all_path]
        filename_count = { filename:filename_list.count(filename) for filename in filename_list if filename_list.count(filename) > 1}
        move_list = [path for path in all_path if DirectoryProcessor.decompose_path(path)[1] in list(filename_count.keys())]
        for src in move_list:
            dir, filename = DirectoryProcessor.path_up(src)
            dst = os.path.join(dir, "MATCHED", filename)
            DirectoryProcessor.move_file(src,dst)
