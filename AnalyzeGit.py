import json
import time
import git
import pickle
import sys
import statistics
import re
from typing import Any, Iterator, List, Optional
from pathlib import Path


file_name = 'myfile.pkl'
def read_history():
    sys.getrecursionlimit()
    sys.setrecursionlimit(10000)
    repo = git.Repo("./", odbt=git.GitCmdObjectDB)
    root = repo.tree()
    queue = []
    queue.append(root)

    start_time = time.time()
    file_info = {}
    try:
        while len(queue) > 0:
            if len(file_info) % 100 == 0:
                print(f"{len(file_info)} files processed")
            curr = queue.pop()
            for blob in curr:
                if(blob.type == 'blob'):
                    commit = next(repo.iter_commits(paths=blob.path, max_count=1))
                    # print(blob.path, commit.committed_date, blob.type, commit.author)
                    file_info[blob.hexsha] = {'blob': blob.path,
                                            'author': commit.author,
                                            "last_edited": commit.committed_datetime,
                                            "test": commit.hexsha
                                            }
            for tree in curr.trees:
                queue.append(tree)
    except Exception as e:
        print("CAUGHT EXCEPTION" , e)
        pass

    print("WRITING TO FILE")

    with open('myfile.pkl', 'wb') as f:
        pickle.dump(file_info, f)

    end_time = time.time()

    print(end_time - start_time)

def read_history_from_file(file_name):
    with open(file_name, 'rb') as f:
      data = pickle.load(f)
      return data

def read_code_owners():
    file1 = open('CODEOWNERS', 'r')
    lines = file1.readlines()
    owners = {}
    for line in lines:
        if line.startswith('#'):
            continue
        if line == '\n':
            continue
        ownerLine = line.split(' ')
        owner = ownerLine[0].strip()
        if owner.startswith('/'):
            owner = owner[1:]
        owners[owner] = ownerLine[1:]
    return owners


class PeekableIterator(Iterator[str]):
    def __init__(self, val: str) -> None:
        self._val = val
        self._idx = -1

    def peek(self) -> Optional[str]:
        if self._idx + 1 >= len(self._val):
            return None
        return self._val[self._idx + 1]

    def __iter__(self) -> "PeekableIterator":
        return self

    def __next__(self) -> str:
        rc = self.peek()
        if rc is None:
            raise StopIteration
        self._idx += 1
        return rc

def patterns_to_regex(allowed_patterns: List[str]) -> Any:
    """
    pattern is glob-like, i.e. the only special sequences it has are:
      - ? - matches single character
      - * - matches any non-folder separator characters
      - ** - matches any characters
      Assuming that patterns are free of braces and backslashes
      the only character that needs to be escaped are dot and plus
    """
    rc = "("
    for idx, pattern in enumerate(allowed_patterns):
        if idx > 0:
            rc += "|"
        pattern_ = PeekableIterator(pattern)
        assert not any(c in pattern for c in "{}()[]\\")
        for c in pattern_:
            if c == ".":
                rc += "\\."
            elif c == "+":
                rc += "\\+"
            elif c == "*":
                if pattern_.peek() == "*":
                    next(pattern_)
                    rc += ".+"
                else:
                    rc += "[^/]+"
            else:
                rc += c
    rc += ")"
    return re.compile(rc)

data = read_history_from_file(file_name)


# Total Files Tracked
num_files = len(data)
print(num_files)

curr_time = time.time()
time_diffs = []
for file in data:
    time_diff = curr_time - data[file]['last_edited'].timestamp()
    data[file]['time_diff'] = time_diff
    time_diffs.append(time_diff)

avg = statistics.mean(time_diffs)
median = statistics.median(time_diffs)

seconds_in_day = 60*60*24

print("Average code age (days) is ", avg / (seconds_in_day))
print("Median code age (days) is", median/ (seconds_in_day))

owners = read_code_owners()
matched_files = {}
owners_regex = {}
for owner in owners:
    owners_regex[owner] = patterns_to_regex([owner])
i = 0
for file in data:
    file_path = data[file]['blob']
    for owner in owners:
        if owners_regex[owner].match(file_path):
            i += 1
            if owner not in matched_files:
                matched_files[owner] = [file]
            else:
                matched_files[owner].append(file)
            break
# for owner in matched_files:
#     print(f'OWNER: {owner} , NUM FILES: {len(matched_files[owner])}')
# print('total matched files', i)

rules = []

with open('.github/merge_rules.json') as fp:
    rules = json.load(fp)

print(rules)

matched_patterns = {}
pattern_rule = {}
for rule in rules:
    pattern_rule[rule['name']] = patterns_to_regex(rule['patterns'])

matched_cnt = 0
for file in data:
    for pattern in pattern_rule:
        if pattern_rule[pattern].match(data[file]['blob']):
            matched_cnt += 1
            if pattern not in matched_patterns:
                matched_patterns[pattern] = [file]
            else:
                matched_patterns[pattern].append(file)
            break
print('num matched:',matched_cnt - len(matched_patterns['superuser']))
for pattern in matched_patterns:
    print(f'pattern: {pattern}, files_matched: {len(matched_patterns[pattern])}')