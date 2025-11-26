import os
import yaml
import subprocess
from difflib import unified_diff


class LoaderYaml:
    @staticmethod
    def join_path(loader, node):
        seq = loader.construct_sequence(node)
        return os.path.join(*[s for s in seq])

    def __init__(self, yml):
        self.yml = yml

    def read_yml(self):
        yaml.add_constructor('!join', self.join_path)
        with open(self.yml, 'r', encoding="utf-8") as file:
            return yaml.full_load(file)


class CommitAuto:
    def __init__(self, yml):
        self.config = yml
        self.backup = f"{self.config}.bak"
        self.history = "change_history.txt"

    @staticmethod
    def read_file(filepath):
        with open(filepath, 'r') as file:
            return file.readlines()

    @staticmethod
    def write_file(filepath, lines):
        with open(filepath, 'w') as file:
            file.writelines(lines)

    @staticmethod
    def run_command(command):
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error running command: {command}")
            print(result.stdout)
            print(result.stderr)
            exit(1)
        return result.stdout.strip()

    @staticmethod
    def get_diff_lines(old_lines, new_lines):
        diff = list(unified_diff(old_lines, new_lines, lineterm=''))
        return [line for line in diff if line.startswith(('+', '-')) and not line.startswith(('+++', '---'))]

    @staticmethod
    def generate_commit_message(diff_lines):
        changes = []
        for i in range(1, len(diff_lines), 2):
            old_value = diff_lines[i - 1].lstrip('- ')
            new_value = diff_lines[i].lstrip('+ ')
            changes.append(f"from {old_value} to {new_value}")
        return "; ".join(changes).replace("\n", "")

    def update_change_history(self, commit_message):
        if not os.path.exists(self.history):
            self.write_file(self.history, [f"1 {commit_message}\n"])
        else:
            lines = self.read_file(self.history)
            if lines:
                last_line = lines[-1]
                last_num = int(last_line.split()[0])
                new_num = last_num + 1
            else:
                new_num = 1
            lines.append(f"{new_num} {commit_message}\n")
            self.write_file(self.history, lines)

    def autocommit(self):
        if os.path.exists(self.backup):
            old_lines = self.read_file(self.backup)
        else:
            old_lines = []
        new_lines = self.read_file(self.config)
        diff_lines = self.get_diff_lines(old_lines, new_lines)

        if not diff_lines:
            print("No changes detected in the config file.")
            return
        commit_message = self.generate_commit_message(diff_lines)
        self.write_file(self.backup, new_lines)
        # self.run_command('ruff check --fix')

        self.run_command(f'git add .')
        self.run_command(f'git commit -m "{commit_message}"')
        self.update_change_history(commit_message)

