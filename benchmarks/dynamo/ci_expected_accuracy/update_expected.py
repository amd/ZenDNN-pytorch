from rockset import RocksetClient
repo = "pytorch/pytorch"
sha = "9f3c5b11b04e948c5f2c1776bb3f268272da2ded" 
def query_job_sha(repo, sha):
    rs = RocksetClient(api_key='Cw9SkltQM0HgUb3qcOYZns9h6nEo39UegI3VyWrgELOWRjBk4Mj5efJWwXNcFYyc',
            host='https://api.usw2a1.rockset.com')

    params = list()
    params.append({"name": "repo", "type": "string", "value": repo})
    params.append({"name": "sha", "type": "string", "value": sha})

    response = rs.QueryLambdas.execute_query_lambda(
        query_lambda='commit_jobs_query',
        version='cc524c5036e78794',
        workspace='commons',
        parameters=params
    )
    return response.results

# print(response.results[0].keys())
# dict_keys(
# ['sha', 'workflowName', 'jobName', 'name', 'id', 
#  'workflowId', 'githubArtifactUrl', 'conclusion',
#  'htmlUrl', 'logUrl', 'durationS', 'queueTimeS',
#  'failureLine', 'failureLineNumber', 'failureCaptures',
#  'time'])


# print("\n".join([f"{k}: {v}" for k, v in response.results[0].items()]))
# sha: 9f3c5b11b04e948c5f2c1776bb3f268272da2ded
# workflowName: Check Labels
# jobName: Check labels
# name: Check Labels / Check labels
# id: 11882825005
# workflowId: 4375467799
# githubArtifactUrl: https://api.github.com/repos/pytorch/pytorch/actions/runs/4375467799/artifacts
# conclusion: success
# htmlUrl: https://github.com/pytorch/pytorch/actions/runs/4375467799/jobs/7656218198
# logUrl: https://ossci-raw-job-status.s3.amazonaws.com/log/11882825005
# durationS: 28
# queueTimeS: -56
# failureLine: None
# failureLineNumber: None
# failureCaptures: None
# time: 2023-03-09T14:28:07.455000Z


def parse_job_name(job_str):
    return (part.strip() for part in job_str.split("/"))

def parse_test_str(test_str):
    return (part.strip() for part in test_str[6:].strip(')k').split(','))

S3_BASE_URL = "https://gha-artifacts.s3.amazonaws.com"

def get_artifacts_urls(results, suites):
    urls = {}
    for r in results:
        config_str, test_str = parse_job_name(r["jobName"])
        if "inductor" == r['workflowName'] and f"test" in r['jobName']:
            suite, shard_id, num_shards, machine = parse_test_str(test_str)
            workflowId = r["workflowId"]
            id = r["id"]
            runattempt = 1 # ? guessing here

            if suite in suites:
                artifact_filename = f"test-reports-test-{suite}-{shard_id}-{num_shards}-{machine}_{id}.zip"
                s3_url = f"{S3_BASE_URL}/{repo}/{workflowId}/{runattempt}/artifact/{artifact_filename}"
                urls[(suite, int(shard_id))] = s3_url
                print(f"{suite} {shard_id}, {num_shards}: {s3_url}")
    return urls

def download_artifacts_and_extract_csvs(urls):
    from io import BytesIO
    from zipfile import ZipFile
    from urllib.request import urlopen

    for (suite, shard), url in urls.items():
        resp = urlopen(url)
        artifact = ZipFile(BytesIO(resp.read()))
        breakpoint()
        for line in artifact.open(file).readlines():
            print(line.decode('utf-8'))

results = query_job_sha(repo, sha)
suites = {
    "inductor_huggingface",
    "inductor_huggingface_dynamic",
    "inductor_timm",
    "inductor_timm_dynamic",
    "inductor_torchbench",
    "inductor_torchbench_dynamic",
}
urls = get_artifacts_urls(results, suites)
csvs = get_artifacts_csvs(urls)

breakpoint()



"""

# S3 prefix:pytorch/pytorch/4375477672/1/artifact
# S3 url https://gha-artifacts.s3.amazonaws.com/pytorch/pytorch/
#               4375477672/1/artifact/test-reports-test-inductor_timm-1-2-linux.g5.4xlarge.nvidia.gpu_11883613183.zip
        https://gha-artifacts.s3.amazonaws.com/pytorch/pytorch/
        4375477672/1/artifact/test-reports-test-inductor_timm-1-2-linux.g5.4xlarge.nvidia.gpu_11883613183.zip
        https://gha-artifacts.s3.amazonaws.com/pytorch/pytorch//
        4375477672/1/artifact/test-reports-test-inductor_timm-1-2-linux.g5.4xlarge.nvidia.gpu_11883613183.zip
        https://gha-artifacts.s3.amazonaws.com/pytorch/pytorch//
        4375477672/1/artifact/test-reports-test-inductor_huggingface-1-1-linux.g5.4xlarge.nvidia.gpu_11883612915.zip

        https://gha-artifacts.s3.amazonaws.com/pytorch/pytorch//
        4375477672/1/artifact/test-reports-test-inductor_torchbench_smoketest_perf-1-1-linux.gcp.a100_11883591742.zip
inductor_torchbench_smoketest_perf  1,  1: https://api.github.com/repos/pytorch/pytorch/actions/runs/4375477672/artifacts
inductor  1,  1: https://api.github.com/repos/pytorch/pytorch/actions/runs/4375477672/artifacts
inductor_distributed  1,  1: https://api.github.com/repos/pytorch/pytorch/actions/runs/4375477672/artifacts
inductor_huggingface  1,  1: https://api.github.com/repos/pytorch/pytorch/actions/runs/4375477672/artifacts
inductor_huggingface_dynamic  1,  1: https://api.github.com/repos/pytorch/pytorch/actions/runs/4375477672/artifacts
inductor_timm  1,  2: https://api.github.com/repos/pytorch/pytorch/actions/runs/4375477672/artifacts
inductor_timm  2,  2: https://api.github.com/repos/pytorch/pytorch/actions/runs/4375477672/artifacts
inductor_timm_dynamic  1,  2: https://api.github.com/repos/pytorch/pytorch/actions/runs/4375477672/artifacts
inductor_timm_dynamic  2,  2: https://api.github.com/repos/pytorch/pytorch/actions/runs/4375477672/artifacts
inductor_torchbench  1,  1: https://api.github.com/repos/pytorch/pytorch/actions/runs/4375477672/artifacts
inductor_torchbench_dynamic  1,  1: https://api.github.com/repos/pytorch/pytorch/actions/runs/4375477672/artifacts
inductor_huggingface_cpu_accuracy  1,  1: https://api.github.com/repos/pytorch/pytorch/actions/runs/4375477672/artifacts
inductor_timm_cpu_accuracy  1,  2: https://api.github.com/repos/pytorch/pytorch/actions/runs/4375477672/artifacts
inductor_timm_cpu_accuracy  2,  2: https://api.github.com/repos/pytorch/pytorch/actions/runs/4375477672/artifacts
inductor_torchbench_cpu_accuracy  1,  1: https://api.github.com/repos/pytorch/pytorch/actions/runs/4375477672/artifacts

"""


"""

{
  "total_count": 3,
  "artifacts": [
    {
      "id": 591322188,
      "node_id": "MDg6QXJ0aWZhY3Q1OTEzMjIxODg=",
      "name": "test-jsons-runattempt1-test-inductor_torchbench_smoketest_perf-1-1-linux.gcp.a100_11883591742.zip",
      "size_in_bytes": 300267,
      "url": "https://api.github.com/repos/pytorch/pytorch/actions/artifacts/591322188",
      "archive_download_url": "https://api.github.com/repos/pytorch/pytorch/actions/artifacts/591322188/zip",
      "expired": false,
      "created_at": "2023-03-09T17:12:28Z",
      "updated_at": "2023-03-09T17:12:29Z",
      "expires_at": "2023-03-23T15:13:49Z",
      "workflow_run": {
        "id": 4375477672,
        "repository_id": 65600975,
        "head_repository_id": 65600975,
        "head_branch": "ciflow/inductor/96346",
        "head_sha": "9f3c5b11b04e948c5f2c1776bb3f268272da2ded"
      }
    },
    {
      "id": 591322189,
      "node_id": "MDg6QXJ0aWZhY3Q1OTEzMjIxODk=",
      "name": "test-reports-runattempt1-test-inductor_torchbench_smoketest_perf-1-1-linux.gcp.a100_11883591742.zip",
      "size_in_bytes": 2287,
      "url": "https://api.github.com/repos/pytorch/pytorch/actions/artifacts/591322189",
      "archive_download_url": "https://api.github.com/repos/pytorch/pytorch/actions/artifacts/591322189/zip",
      "expired": false,
      "created_at": "2023-03-09T17:12:28Z",
      "updated_at": "2023-03-09T17:12:30Z",
      "expires_at": "2023-03-23T15:13:50Z",
      "workflow_run": {
        "id": 4375477672,
        "repository_id": 65600975,
        "head_repository_id": 65600975,
        "head_branch": "ciflow/inductor/96346",
        "head_sha": "9f3c5b11b04e948c5f2c1776bb3f268272da2ded"
      }
    },
    {
      "id": 591322190,
      "node_id": "MDg6QXJ0aWZhY3Q1OTEzMjIxOTA=",
      "name": "usage-log-runattempt1-test-inductor_torchbench_smoketest_perf-1-1-linux.gcp.a100_11883591742.zip",
      "size_in_bytes": 1892650,
      "url": "https://api.github.com/repos/pytorch/pytorch/actions/artifacts/591322190",
      "archive_download_url": "https://api.github.com/repos/pytorch/pytorch/actions/artifacts/591322190/zip",
      "expired": false,
      "created_at": "2023-03-09T17:12:28Z",
      "updated_at": "2023-03-09T17:12:30Z",
      "expires_at": "2023-03-23T15:13:51Z",
      "workflow_run": {
        "id": 4375477672,
        "repository_id": 65600975,
        "head_repository_id": 65600975,
        "head_branch": "ciflow/inductor/96346",
        "head_sha": "9f3c5b11b04e948c5f2c1776bb3f268272da2ded"
      }
    }
  ]
}

"""
breakpoint()
# print(response.results)