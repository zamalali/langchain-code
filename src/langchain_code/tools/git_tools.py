# git_tools.py
"""
GitHub tools implemented in the recommended LangChain format using the @tool decorator.

Prereqs:
    pip install -U pygithub langchain-core

Environment:
    GITHUB_TOKEN  -> Personal Access Token if 'token' arg is not passed

Notes:
- "apply" flags default to False (DRY_RUN). Set apply=True to actually write/modify.
- All tools raise RuntimeError on API errors with a helpful message for the agent.
"""

from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field
from langchain_core.tools import tool

try:
    from github import Github, GithubException, ContentFile
except Exception as e:  # pragma: no cover
    raise RuntimeError("PyGithub is required. Install with `pip install PyGithub`.") from e


# -------------------------------
# Internal helpers
# -------------------------------
def _gh(token: Optional[str]) -> Github:
    tok = token or os.getenv("GITHUB_API_KEY")
    if not tok:
        raise RuntimeError("Missing GitHub token. Set GITHUB_API_KEY or pass `token`.")
    return Github(tok, per_page=100)

def _repo(token: Optional[str], repo_full_name: str):
    try:
        return _gh(token).get_repo(repo_full_name)
    except GithubException as e:
        msg = getattr(e, "data", None) or str(e)
        raise RuntimeError(f"Repo '{repo_full_name}' not found or access denied: {msg}") from e

def _dry(apply: bool, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not apply:
        return {"status": "DRY_RUN", **payload}
    return None


# -------------------------------
# Tools
# -------------------------------
@tool("github_list_branches")
def github_list_branches(repo_full_name: str, token: Optional[str] = None) -> Dict[str, Any]:
    """List branches in a repository and indicate the default branch."""
    repo = _repo(token, repo_full_name)
    branches = [b.name for b in repo.get_branches()]
    return {"default_branch": repo.default_branch, "branches": branches}


@tool("github_list_files")
def github_list_files(
    repo_full_name: str,
    ref: str = "main",
    path: str = "",
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """Recursively list files under a path at a given ref/branch."""
    repo = _repo(token, repo_full_name)
    queue = [path or ""]
    files: List[str] = []
    while queue:
        p = queue.pop(0)
        try:
            contents = repo.get_contents(p, ref=ref)
        except GithubException as e:
            msg = getattr(e, "data", None) or str(e)
            raise RuntimeError(f"Failed to list '{p}' at {ref}: {msg}") from e
        if isinstance(contents, list):
            for c in contents:
                if c.type == "dir":
                    queue.append(c.path)
                elif c.type == "file":
                    files.append(c.path)
        else:
            if contents.type == "file":
                files.append(contents.path)
    return {"ref": ref, "path": path or "", "files": files}


@tool("github_read_file")
def github_read_file(
    repo_full_name: str,
    path: str,
    ref: str = "main",
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """Read a UTF-8 text file at a given ref/branch."""
    repo = _repo(token, repo_full_name)
    try:
        cf = repo.get_contents(path, ref=ref)
        content = cf.decoded_content.decode("utf-8", errors="replace")
        return {"path": path, "ref": ref, "encoding": "utf-8", "content": content}
    except GithubException as e:
        msg = getattr(e, "data", None) or str(e)
        raise RuntimeError(f"Read failed for '{path}' at {ref}: {msg}") from e


@tool("github_write_file")
def github_write_file(
    repo_full_name: str,
    path: str,
    content: str,
    message: str,
    ref: str = "main",
    create_if_missing: bool = True,
    apply: bool = False,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """Create/update a file on a branch with a commit message. Respects 'apply' flag."""
    payload = _dry(apply, {"action": "write_file", "path": path, "ref": ref, "message": message})
    if payload:
        return payload
    repo = _repo(token, repo_full_name)
    try:
        try:
            existing = repo.get_contents(path, ref=ref)
            res = repo.update_file(
                path=path, message=message, content=content, sha=existing.sha, branch=ref
            )
            status = "updated"
        except GithubException:
            if not create_if_missing:
                raise
            res = repo.create_file(path=path, message=message, content=content, branch=ref)
            status = "created"
        commit = res.get("commit")
        sha = getattr(commit, "sha", None) if commit else None
        return {"status": status, "path": path, "ref": ref, "commit": sha}
    except GithubException as e:
        msg = getattr(e, "data", None) or str(e)
        raise RuntimeError(f"Write failed for '{path}' at {ref}: {msg}") from e


@tool("github_create_branch")
def github_create_branch(
    repo_full_name: str,
    new_branch: str,
    from_ref: str = "main",
    apply: bool = False,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a new branch from an existing ref/branch. Respects 'apply' flag."""
    payload = _dry(apply, {"action": "create_branch", "new_branch": new_branch, "from_ref": from_ref})
    if payload:
        return payload
    repo = _repo(token, repo_full_name)
    try:
        base_ref = repo.get_git_ref(f"heads/{from_ref}")
        repo.create_git_ref(ref=f"refs/heads/{new_branch}", sha=base_ref.object.sha)
        return {"status": "created", "new_branch": new_branch, "from_ref": from_ref}
    except GithubException as e:
        msg = getattr(e, "data", None) or str(e)
        raise RuntimeError(f"Create branch failed: {msg}") from e


@tool("github_compare_branches")
def github_compare_branches(
    repo_full_name: str,
    base: str,
    head: str,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """Show diff summary between base and head branches (file stats, commit counts)."""
    repo = _repo(token, repo_full_name)
    try:
        comp = repo.compare(base, head)
    except GithubException as e:
        msg = getattr(e, "data", None) or str(e)
        raise RuntimeError(f"Compare failed: {msg}") from e
    files = [
        {
            "filename": f.filename,
            "status": f.status,
            "additions": f.additions,
            "deletions": f.deletions,
            "changes": f.changes,
        }
        for f in comp.files
    ]
    return {
        "ahead_by": comp.ahead_by,
        "behind_by": comp.behind_by,
        "total_commits": comp.total_commits,
        "files": files,
    }


@tool("github_create_pull_request")
def github_create_pull_request(
    repo_full_name: str,
    base: str,
    head: str,
    title: str,
    body: str = "",
    draft: bool = False,
    apply: bool = False,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """Open a pull request (supports draft PRs). Respects 'apply' flag."""
    payload = _dry(apply, {"action": "create_pr", "base": base, "head": head, "title": title, "draft": draft})
    if payload:
        return payload
    repo = _repo(token, repo_full_name)
    try:
        pr = repo.create_pull(title=title, body=body or "", base=base, head=head, draft=draft)
        return {"status": "created", "number": pr.number, "html_url": pr.html_url, "draft": pr.draft}
    except GithubException as e:
        msg = getattr(e, "data", None) or str(e)
        raise RuntimeError(f"Create PR failed: {msg}") from e


@tool("github_merge_pull_request")
def github_merge_pull_request(
    repo_full_name: str,
    number: int,
    merge_method: str = "squash",  # 'merge' | 'squash' | 'rebase'
    commit_title: Optional[str] = None,
    commit_message: Optional[str] = None,
    apply: bool = False,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """Merge a PR with merge/squash/rebase. Respects 'apply' flag."""
    if merge_method not in {"merge", "squash", "rebase"}:
        raise RuntimeError("merge_method must be one of: merge, squash, rebase")
    payload = _dry(apply, {"action": "merge_pr", "number": number, "merge_method": merge_method})
    if payload:
        return payload
    repo = _repo(token, repo_full_name)
    pr = repo.get_pull(number)
    try:
        res = pr.merge(
            merge_method=merge_method,
            commit_title=commit_title,
            commit_message=commit_message,
        )
        return {"merged": res.merged, "message": res.message, "sha": getattr(res, "sha", None)}
    except GithubException as e:
        msg = getattr(e, "data", None) or str(e)
        raise RuntimeError(f"Merge failed: {msg}") from e


@tool("github_comment_on_pr")
def github_comment_on_pr(
    repo_full_name: str,
    number: int,
    body: str,
    apply: bool = False,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """Add a general (issue-style) comment to a PR. Respects 'apply' flag."""
    payload = _dry(apply, {"action": "comment_pr", "number": number})
    if payload:
        return payload
    repo = _repo(token, repo_full_name)
    pr = repo.get_pull(number)
    try:
        c = pr.create_issue_comment(body)
        return {"status": "commented", "id": c.id, "body": c.body}
    except GithubException as e:
        msg = getattr(e, "data", None) or str(e)
        raise RuntimeError(f"PR comment failed: {msg}") from e


@tool("github_list_pull_requests")
def github_list_pull_requests(
    repo_full_name: str,
    state: str = "open",  # open|closed|all
    head: Optional[str] = None,
    base: Optional[str] = None,
    label: Optional[str] = None,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """List PRs with optional filters (state/base/head/label)."""
    repo = _repo(token, repo_full_name)
    pulls = repo.get_pulls(state=state)
    results = []
    for pr in pulls:
        if base and pr.base.ref != base:
            continue
        if head and (pr.head.label != head and pr.head.ref != head):
            continue
        if label:
            labels = {l.name for l in pr.get_labels()}
            if label not in labels:
                continue
        results.append(
            {
                "number": pr.number,
                "title": pr.title,
                "state": pr.state,
                "draft": pr.draft,
                "base": pr.base.ref,
                "head": pr.head.ref,
                "user": pr.user.login if pr.user else None,
                "html_url": pr.html_url,
            }
        )
    return {"count": len(results), "pulls": results}


@tool("github_create_issue")
def github_create_issue(
    repo_full_name: str,
    title: str,
    body: str = "",
    labels: Optional[List[str]] = None,
    apply: bool = False,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """Create an issue with optional body and labels. Respects 'apply' flag."""
    payload = _dry(apply, {"action": "create_issue", "title": title})
    if payload:
        return payload
    repo = _repo(token, repo_full_name)
    try:
        issue = repo.create_issue(title=title, body=body or "", labels=labels or [])
        return {"status": "created", "number": issue.number, "html_url": issue.html_url}
    except GithubException as e:
        msg = getattr(e, "data", None) or str(e)
        raise RuntimeError(f"Create issue failed: {msg}") from e


@tool("github_comment_on_issue")
def github_comment_on_issue(
    repo_full_name: str,
    number: int,
    body: str,
    apply: bool = False,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """Add a comment to an issue. Respects 'apply' flag."""
    payload = _dry(apply, {"action": "comment_issue", "number": number})
    if payload:
        return payload
    repo = _repo(token, repo_full_name)
    issue = repo.get_issue(number=number)
    try:
        c = issue.create_comment(body)
        return {"status": "commented", "id": c.id, "body": c.body}
    except GithubException as e:
        msg = getattr(e, "data", None) or str(e)
        raise RuntimeError(f"Issue comment failed: {msg}") from e


@tool("github_add_labels")
def github_add_labels(
    repo_full_name: str,
    number: int,
    labels: List[str],
    apply: bool = False,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """Add labels to an issue or PR by number. Respects 'apply' flag."""
    payload = _dry(apply, {"action": "add_labels", "number": number, "labels": labels})
    if payload:
        return payload
    repo = _repo(token, repo_full_name)
    issue = repo.get_issue(number=number)  # works for PRs too
    try:
        issue.add_to_labels(*labels)
        return {"status": "labeled", "number": number, "labels": labels}
    except GithubException as e:
        msg = getattr(e, "data", None) or str(e)
        raise RuntimeError(f"Add labels failed: {msg}") from e


@tool("github_list_commits")
def github_list_commits(
    repo_full_name: str,
    branch: str = "main",
    limit: int = 20,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """List recent commits on a branch with SHA, message, author, date, and URL."""
    repo = _repo(token, repo_full_name)
    commits = repo.get_commits(sha=branch)
    out: List[Dict[str, Any]] = []
    for i, c in enumerate(commits):
        if i >= limit:
            break
        out.append(
            {
                "sha": c.sha,
                "message": c.commit.message,
                "author": getattr(c.author, "login", None),
                "date": c.commit.author.date.isoformat(),
                "html_url": c.html_url,
            }
        )
    return {"branch": branch, "commits": out}


@tool("github_close_item")
def github_close_item(
    repo_full_name: str,
    number: int,
    kind: Optional[str] = None,  # 'pr' or 'issue' (auto-detect if omitted)
    apply: bool = False,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """Close a PR or Issue by number (auto-detects if not specified). Respects 'apply' flag."""
    payload = _dry(apply, {"action": "close_item", "number": number, "kind": kind or "auto"})
    if payload:
        return payload
    repo = _repo(token, repo_full_name)
    # Prefer PR if exists
    pr_obj = None
    try:
        pr_obj = repo.get_pull(number)
        if pr_obj is not None and pr_obj.number == number:
            kind = kind or "pr"
    except GithubException:
        pass
    try:
        if kind == "pr":
            pr = pr_obj or repo.get_pull(number)
            pr.edit(state="closed")
            return {"status": "closed", "type": "pr", "number": number}
        else:
            issue = repo.get_issue(number=number)
            issue.edit(state="closed")
            return {"status": "closed", "type": "issue", "number": number}
    except GithubException as e:
        msg = getattr(e, "data", None) or str(e)
        raise RuntimeError(f"Close failed: {msg}") from e


@tool("github_get_readme")
def github_get_readme(
    repo_full_name: str,
    ref: Optional[str] = None,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """Fetch README content as UTF-8 text from default branch or a specific ref."""
    repo = _repo(token, repo_full_name)
    try:
        cf: ContentFile.ContentFile = repo.get_readme(ref=ref) if ref else repo.get_readme()
        content = cf.decoded_content.decode("utf-8", errors="replace")
        return {"path": cf.path, "ref": ref or repo.default_branch, "content": content}
    except GithubException as e:
        msg = getattr(e, "data", None) or str(e)
        raise RuntimeError(f"README fetch failed: {msg}") from e


# -------------------------------
# Toolkit convenience
# -------------------------------
def make_github_tools() -> List[Any]:
    """
    Return all GitHub tools defined in this module.

    Usage:
        from langgraph.prebuilt import create_react_agent
        from langchain.chat_models import init_chat_model

        llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
        tools = make_github_tools()
        agent = create_react_agent(llm, tools)

        # Example:
        # events = agent.stream(
        #   {"messages": [("user", "List branches for owner/repo")]}
        # )
    """
    return [
        github_list_branches,
        github_list_files,
        github_read_file,
        github_write_file,
        github_create_branch,
        github_compare_branches,
        github_create_pull_request,
        github_merge_pull_request,
        github_comment_on_pr,
        github_list_pull_requests,
        github_create_issue,
        github_comment_on_issue,
        github_add_labels,
        github_list_commits,
        github_close_item,
        github_get_readme,
    ]
