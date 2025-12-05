#!/usr/bin/env python3
"""
SearchAI Goodrich - Resource Verification Agent
A tool to find and verify academic resources for research subjects.
"""

import os
import json
import sys
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse
from openai import OpenAI
import anthropic
from anthropic import APIError
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Suppress Anthropics model deprecation warnings to avoid noisy CLI output
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"anthropic.*")
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r".*model .* is deprecated and will reach end-of-life.*",
)

CLAUDE_MODEL_PREFERENCE = [
    model.strip()
    for model in os.getenv(
        "ANTHROPIC_MODELS",
        "claude-4.5-sonnet,"
        "claude-3.5-sonnet-20241022,"
        "claude-3.5-sonnet,"
        "claude-3-sonnet-20240229,"
        "claude-2.1",
    ).split(",")
    if model.strip()
]


class ClaudeModelNotFound(Exception):
    """Raised when a requested Claude model is unavailable."""



def get_openai_client():
    """Initialize OpenAI client"""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        sys.exit(1)
    return OpenAI(api_key=api_key)

def get_claude_client():
    """Initialize Claude client"""
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)
    return anthropic.Anthropic(api_key=api_key)

def create_output_directory(name):
    """Create output directory structure"""
    output_dir = Path(f"outputs/{name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_json(path: Path, payload: Any) -> None:
    """Persist JSON payload with UTF-8 encoding."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def quick_url_filter(urls: List[str]) -> List[str]:
    """Quickly reject URLs with obviously non-academic patterns."""
    rejected_patterns = [
        "wikipedia.org",
        "youtube.com",
        "blogspot.",
        "wordpress.com",
        "medium.com",
        "quora.com",
        ".com/blog/",
        "news.",
        "article/abstract",
    ]
    return [url for url in urls if url and not any(p in url for p in rejected_patterns)]


def apply_prevalidation_filter(
    sources: List[Dict[str, Any]], label: str
) -> Tuple[List[Dict[str, Any]], int]:
    """Filter source dictionaries using the quick URL filter for early triage."""
    allowed_urls = set(quick_url_filter([src.get("url", "") for src in sources]))
    filtered: List[Dict[str, Any]] = []
    rejected_count = 0
    for source in sources:
        url = source.get("url", "")
        if url and url in allowed_urls:
            filtered.append(source)
        else:
            rejected_count += 1
    if rejected_count:
        print(
            f"   - Quick filter removed {rejected_count} {label} sources as obvious non-starters"
        )
    return filtered, rejected_count

def extract_openai_text(response) -> str:
    """Extract concatenated text from a Responses API result."""
    texts: List[str] = []
    for item in getattr(response, "output", []) or []:
        # Each item is typically a Message with a content list
        for content in getattr(item, "content", []) or []:
            if getattr(content, "type", None) == "text":
                text = getattr(content, "text", "")
                if text:
                    texts.append(text)
    combined = "\n".join(texts).strip()
    if not combined:
        raise ValueError("OpenAI response did not contain text content")
    return combined


def call_openai_json(client: OpenAI, system_prompt: str, user_prompt: str, *, expect_array: bool = False) -> Any:
    """Call GPT-5.1 (Responses when available, chat completions otherwise) and parse JSON."""
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            has_responses = hasattr(client, "responses") and hasattr(client.responses, "create")
            if has_responses:
                response = client.responses.create(
                    model="gpt-5.1",
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.1,
                )
                text = extract_openai_text(response)
            else:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                response = client.chat.completions.create(
                    model="gpt-5.1",
                    messages=messages,
                    temperature=0.1,
                )
                text = response.choices[0].message.content
            data = json.loads(text)
            if expect_array and not isinstance(data, list):
                raise ValueError("Expected JSON array from OpenAI response")
            return data
        except (json.JSONDecodeError, ValueError, RuntimeError) as exc:
            last_error = exc
        except Exception as exc:  # pragma: no cover - defensive
            last_error = exc
        time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to obtain valid JSON from OpenAI: {last_error}")


def get_openai_sources(client: OpenAI, name: str) -> List[Dict[str, Any]]:
    """Get initial 20 verified academic sources from OpenAI"""
    prompt = f"""
MISSION: You are a precision retrieval agent. Your sole task is to find valid, verifiable,
scholarly online resources about a given historical thinker. Your outputs will directly
populate a trusted knowledge base for students. Accuracy and adherence to these rules
are paramount.

1. DOMAIN & SOURCE VERIFICATION
• Primary Trusted Domains: .edu, .gov, .org (only academic, museum, or research
institutes), .ac.*, and specific scholarly repositories.
• Approved Repositories: archive.org/details/*, jstor.org/stable/, jstor.org/open/,
persee.fr, philpapers.org, brill.com/view/journals (open access), and university
scholarly publishing platforms (e.g., ucpress.edu, oup.com/academic — only if openly
accessible).
• Verification Step: Before returning, you must logically infer that the domain is
legitimately academic. A .org should be a known entity like plato.stanford.edu,
perseids.org, or thebritishmuseum.ac.uk.

2. ACCESSIBILITY & FUNCTIONALITY CRITERIA (NON-NEGOTIABLE)
• Live & Open Access: Every URL must be a live, direct link to the full content,
requiring no login, payment, or institutional subscription.
• Content Must Be Machine-Ingestible: Text must be selectable. If a PDF, it must
be text-based, not a scan of a physical book. (Prioritize HTML pages or true text-PDFs).
• Blocked Categories: No links to publishers' abstract-only pages, library portal
redirects, Google Books previews, Amazon, blogs, WordPress sites, mainstream news
(.com), Wikipedia, Medium, Substack, or any site with dominant ads/pop-ups.

3. CONTENT QUALITY FILTERS
• Prioritized Content Types (in order):
  1. Peer-reviewed journal articles (open access)
  2. Authoritative reference entries (e.g., Stanford Encyclopedia of Philosophy)
  3. Academic monographs or chapters hosted on open repositories
  4. University department publications (lecture series, working papers)
  5. Official translations of primary texts from academic projects
  6. Conference proceedings from recognized societies

• Contextual Relevance: For each thinker, prioritize:
  - Philosophers: Philological analysis, critical editions, commentaries.
  - Economists: Historical economic research, textual criticism of original works.
  - Scientists/Artists: Historical studies from scholarly institutions, digitized notebooks.

4. STRICT PROHIBITIONS
• NO Hallucination: Only return URLs you have verified exist from your knowledge base.
If perfect matches are scarce, return fewer or state none, but do NOT invent.
• NO Non-Scholarly Sources: Exclude: britannica.com, biography.com, sparknotes.com,
gradesaver.com, academia.edu (unless it's a direct, verified upload of a published
paper), researchgate.net (same strict condition), youtube.com, podcasts, and AI-generated
content sites.

5. VALIDATION WORKFLOW (Internal Checklist)
For each potential URL, ask:
- Is the domain from the approved list or verifiably academic?
- Is the content openly accessible without barriers?
- Is the author/source a scholar, university, museum, or research institute?
- Is the content a substantive scholarly work, not a summary or blog post?
- Is the format (HTML/text-PDF) suitable for RAG ingestion?

6. OUTPUT FORMAT
Return a valid JSON array of objects. Nothing else.

[
  {{
    "thinker": "{name}",
    "url": "https://exact.direct.link.to/full-content",
    "source_title": "[Title of the Article, Paper, or Entry]",
    "publisher": "[University Press, Journal Name, or Institution]",
    "content_type": "journal_article | reference_entry | book_chapter | open_access_monograph | primary_text",
    "access_note": "open_access"
  }}
]

If no new, valid sources are found, return an empty array: []

Target Thinker: {name}
Begin search.
"""

    system_prompt = "You are a research assistant that finds only real, verified academic sources. Never hallucinate URLs."
    return call_openai_json(client, system_prompt, prompt, expect_array=True)


def get_openai_additional_sources(client: OpenAI, name: str, existing_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Use OpenAI to generate additional sources when Claude is unavailable."""
    existing_json = json.dumps(existing_results, indent=2)
    prompt = f"""
Find 20 additional verified, fully accessible academic sources about {name}.

IMPORTANT RULES:
1. Do NOT provide any URL that appears in the existing results below:
{existing_json}

2. Every source MUST be different and not overlap with the list above.

MISSION: You are a precision retrieval agent. Your sole task is to find valid, verifiable,
scholarly online resources about a given historical thinker. Your outputs will directly
populate a trusted knowledge base for students. Accuracy and adherence to these rules
are paramount.

1. DOMAIN & SOURCE VERIFICATION
• Primary Trusted Domains: .edu, .gov, .org (only academic, museum, or research
institutes), .ac.*, and specific scholarly repositories.
• Approved Repositories: archive.org/details/*, jstor.org/stable/, jstor.org/open/,
persee.fr, philpapers.org, brill.com/view/journals (open access), and university
scholarly publishing platforms (e.g., ucpress.edu, oup.com/academic — only if openly
accessible).
• Verification Step: Before returning, you must logically infer that the domain is
legitimately academic. A .org should be a known entity like plato.stanford.edu,
perseids.org, or thebritishmuseum.ac.uk.

2. ACCESSIBILITY & FUNCTIONALITY CRITERIA (NON-NEGOTIABLE)
• Live & Open Access: Every URL must be a live, direct link to the full content,
requiring no login, payment, or institutional subscription.
• Content Must Be Machine-Ingestible: Text must be selectable. If a PDF, it must
be text-based, not a scan of a physical book. (Prioritize HTML pages or true text-PDFs).
• Blocked Categories: No links to publishers' abstract-only pages, library portal
redirects, Google Books previews, Amazon, blogs, WordPress sites, mainstream news
(.com), Wikipedia, Medium, Substack, or any site with dominant ads/pop-ups.

3. CONTENT QUALITY FILTERS
• Prioritized Content Types (in order):
  1. Peer-reviewed journal articles (open access)
  2. Authoritative reference entries (e.g., Stanford Encyclopedia of Philosophy)
  3. Academic monographs or chapters hosted on open repositories
  4. University department publications (lecture series, working papers)
  5. Official translations of primary texts from academic projects
  6. Conference proceedings from recognized societies

• Contextual Relevance: For each thinker, prioritize:
  - Philosophers: Philological analysis, critical editions, commentaries.
  - Economists: Historical economic research, textual criticism of original works.
  - Scientists/Artists: Historical studies from scholarly institutions, digitized notebooks.

4. STRICT PROHIBITIONS
• NO Hallucination: Only return URLs you have verified exist from your knowledge base.
If perfect matches are scarce, return fewer or state none, but do NOT invent.
• NO Non-Scholarly Sources: Exclude: britannica.com, biography.com, sparknotes.com,
gradesaver.com, academia.edu (unless it's a direct, verified upload of a published
paper), researchgate.net (same strict condition), youtube.com, podcasts, and AI-generated
content sites.
• NO Duplicates: Cross-check against the provided list of existing URLs to avoid redundancy.

5. VALIDATION WORKFLOW (Internal Checklist)
For each potential URL, ask:
- Is the domain from the approved list or verifiably academic?
- Is the content openly accessible without barriers?
- Is the author/source a scholar, university, museum, or research institute?
- Is the content a substantive scholarly work, not a summary or blog post?
- Is the format (HTML/text-PDF) suitable for RAG ingestion?

6. OUTPUT FORMAT
Return a valid JSON array of objects. Nothing else.

[
  {{
    "thinker": "{name}",
    "url": "https://exact.direct.link.to/full-content",
    "source_title": "[Title of the Article, Paper, or Entry]",
    "publisher": "[University Press, Journal Name, or Institution]",
    "content_type": "journal_article | reference_entry | book_chapter | open_access_monograph | primary_text",
    "access_note": "open_access"
  }}
]

If no new, valid sources are found, return an empty array: []

Target Thinker: {name}
Existing URLs to Exclude: {existing_json}
Begin search.
"""

    system_prompt = "You are a research assistant that finds only real, verified academic sources. Never hallucinate URLs."
    return call_openai_json(client, system_prompt, prompt, expect_array=True)

def request_claude_sources(client, prompt: str, model: str, claude_messages: List[Dict[str, Any]]):
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            try:
                response = client.messages.create(
                    model=model,
                    max_output_tokens=2000,
                    messages=claude_messages,
                )
            except TypeError as err:
                # Older Anthropics SDKs still expect max_tokens
                if "max_tokens" in str(err):
                    response = client.messages.create(
                        model=model,
                        max_tokens=2000,
                        messages=claude_messages,
                    )
                else:
                    raise
            text = response.content[0].text
            data = json.loads(text)
            if not isinstance(data, list):
                raise ValueError("Expected JSON array from Claude response")
            return data
        except APIError as exc:
            if getattr(exc, "status_code", None) == 404:
                raise ClaudeModelNotFound(f"Model '{model}' not available") from exc
            last_error = exc
        except ClaudeModelNotFound:
            raise
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
        except Exception as exc:  # pragma: no cover - defensive
            last_error = exc
        time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to obtain valid JSON from Claude model {model}: {last_error}")


def get_claude_sources(client, name, openai_results):
    """Get additional 20 unique sources from Claude"""
    openai_results_json = json.dumps(openai_results, indent=2)
    prompt = f"""
Find 20 additional verified, fully accessible academic sources about {name}.

IMPORTANT RULES:
1. Do NOT provide any URL that appears in the existing results below:
{openai_results_json}

2. Every source MUST be different and not overlap with the list above.

MISSION: You are a precision retrieval agent. Your sole task is to find valid, verifiable,
scholarly online resources about a given historical thinker. Your outputs will directly
populate a trusted knowledge base for students. Accuracy and adherence to these rules
are paramount.

1. DOMAIN & SOURCE VERIFICATION
• Primary Trusted Domains: .edu, .gov, .org (only academic, museum, or research
institutes), .ac.*, and specific scholarly repositories.
• Approved Repositories: archive.org/details/*, jstor.org/stable/, jstor.org/open/,
persee.fr, philpapers.org, brill.com/view/journals (open access), and university
scholarly publishing platforms (e.g., ucpress.edu, oup.com/academic — only if openly
accessible).
• Verification Step: Before returning, you must logically infer that the domain is
legitimately academic. A .org should be a known entity like plato.stanford.edu,
perseids.org, or thebritishmuseum.ac.uk.

2. ACCESSIBILITY & FUNCTIONALITY CRITERIA (NON-NEGOTIABLE)
• Live & Open Access: Every URL must be a live, direct link to the full content,
requiring no login, payment, or institutional subscription.
• Content Must Be Machine-Ingestible: Text must be selectable. If a PDF, it must
be text-based, not a scan of a physical book. (Prioritize HTML pages or true text-PDFs).
• Blocked Categories: No links to publishers' abstract-only pages, library portal
redirects, Google Books previews, Amazon, blogs, WordPress sites, mainstream news
(.com), Wikipedia, Medium, Substack, or any site with dominant ads/pop-ups.

3. CONTENT QUALITY FILTERS
• Prioritized Content Types (in order):
  1. Peer-reviewed journal articles (open access)
  2. Authoritative reference entries (e.g., Stanford Encyclopedia of Philosophy)
  3. Academic monographs or chapters hosted on open repositories
  4. University department publications (lecture series, working papers)
  5. Official translations of primary texts from academic projects
  6. Conference proceedings from recognized societies

• Contextual Relevance: For each thinker, prioritize:
  - Philosophers: Philological analysis, critical editions, commentaries.
  - Economists: Historical economic research, textual criticism of original works.
  - Scientists/Artists: Historical studies from scholarly institutions, digitized notebooks.

4. STRICT PROHIBITIONS
• NO Hallucination: Only return URLs you have verified exist from your knowledge base.
If perfect matches are scarce, return fewer or state none, but do NOT invent.
• NO Non-Scholarly Sources: Exclude: britannica.com, biography.com, sparknotes.com,
gradesaver.com, academia.edu (unless it's a direct, verified upload of a published
paper), researchgate.net (same strict condition), youtube.com, podcasts, and AI-generated
content sites.
• NO Duplicates: Cross-check against the provided list of existing URLs to avoid redundancy.

5. VALIDATION WORKFLOW (Internal Checklist)
For each potential URL, ask:
- Is the domain from the approved list or verifiably academic?
- Is the content openly accessible without barriers?
- Is the author/source a scholar, university, museum, or research institute?
- Is the content a substantive scholarly work, not a summary or blog post?
- Is the format (HTML/text-PDF) suitable for RAG ingestion?

6. OUTPUT FORMAT
Return a valid JSON array of objects. Nothing else.

[
  {{
    "thinker": "{name}",
    "url": "https://exact.direct.link.to/full-content",
    "source_title": "[Title of the Article, Paper, or Entry]",
    "publisher": "[University Press, Journal Name, or Institution]",
    "content_type": "journal_article | reference_entry | book_chapter | open_access_monograph | primary_text",
    "access_note": "open_access"
  }}
]

If no new, valid sources are found, return an empty array: []

Target Thinker: {name}
Existing URLs to Exclude: {openai_results_json}
Begin search.
"""

    claude_messages: List[Dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                }
            ],
        }
    ]

    last_error: Exception | None = None
    for model in CLAUDE_MODEL_PREFERENCE:
        try:
            return request_claude_sources(client, prompt, model, claude_messages)
        except ClaudeModelNotFound as exc:
            last_error = exc
            continue
        except Exception as exc:
            last_error = exc
            break
    raise RuntimeError(f"Failed to obtain valid JSON from Claude: {last_error}")

def validate_sources(client, name, openai_results, claude_results):
    """Validate all 40 sources using OpenAI as evaluator"""
    prompt = f"""
        CRITICAL VALIDATION MISSION:
You are an academic librarian validating sources for a student knowledge base about {name}.

EVALUATION CRITERIA (ALL MUST PASS):
1. **URL VALIDITY & ACCESSIBILITY**:
   - URL must be real (not hallucinated) and resolve to actual content
   - Must open in standard browsers without login, paywall, or registration
   - No redirects to unrelated pages or "page not found" errors

2. **DOMAIN AUTHORITY** (APPROVED ONLY):
   - Primary: .edu, .gov, .ac.*
   - Academic .org: Must be research institutes, museums, or scholarly projects
   - Verified repositories: archive.org/details/*, jstor.org/stable/, philpapers.org
   - University press/open access journals with clear academic affiliation

3. **CONTENT TYPE** (APPROVED ONLY):
   - Peer-reviewed journal articles (open access)
   - Academic monographs/chapters from verified repositories
   - University publications, research center papers
   - Authoritative reference works (e.g., Stanford Encyclopedia of Philosophy)
   - Conference proceedings from academic societies

4. **FORMAT SUITABILITY**:
   - Text must be selectable/copyable (machine-readable)
   - PDFs must be text-based (not scanned images)
   - HTML pages preferred over PDFs when both available

5. **RELEVANCE TO {name}**:
   - Content must substantially address {name}'s work, philosophy, or historical context
   - Primary source analysis, textual criticism, biographical scholarship
   - Avoid tangential mentions or brief references

STRICT PROHIBITIONS (IMMEDIATE REJECTION):
- Any .com domain (except university presses: cambridge.org, oup.com when open access)
- Wikipedia, Britannica, Quora, Medium, Substack, blogs
- Academia.edu, ResearchGate (unless verified published paper)
- YouTube, podcasts, news articles, book reviews without substantial analysis
- Scanned-image PDFs, paywalled abstracts only, library catalog entries
- Student essays, course syllabi, PowerPoint presentations

SOURCE LISTS TO VALIDATE:

OpenAI Results:
{json.dumps(openai_results, indent=2)}

Claude Results:
{json.dumps(claude_results, indent=2)}

VALIDATION PROCESS:
For EACH source in both lists:
1. Check if URL pattern is plausible (no obviously fake domains)
2. Evaluate domain against approved/prohibited lists above
3. Assess content type based on description/publisher
4. If ANY doubt about accessibility or quality, REJECT

OUTPUT REQUIREMENTS:
Return ONLY valid JSON with this structure:
{{
  "approved": [
    {{
      "url": "https://verified.academic.source/paper",
      "reason": "Peer-reviewed article from university press, open access"
    }}
  ],
  "denied": [
    {{
      "url": "https://suspicious.source/page",
      "description": "Brief summary without analysis",
      "reason": "Non-academic domain (.com) and lacks scholarly depth"
    }}
  ]
}}

IMPORTANT:
- Approve ONLY sources meeting ALL criteria above
- If a source is borderline (e.g., ResearchGate link to published paper), DENY to maintain quality
- Check for duplicates across both lists; if duplicate, include only once in approved
- Be CONSERVATIVE: 1 high-quality source is better than 10 questionable ones
- Maximum 20 approved sources total (prioritize highest quality)

    """

    system_prompt = "You are an uncompromising academic validation system. Your purpose is to ensure only verified, accessible, scholarly sources enter a trusted student knowledge base. You follow strict criteria without exception. You reject any source with accessibility issues, questionable provenance, or insufficient scholarly rigor. You prioritize reliability over quantity."
    return call_openai_json(client, system_prompt, prompt)


def generate_validation_report(
    name: str, approved: List[Dict[str, Any]], denied: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Create a concise summary for human spot-checking of validation output."""
    top_domains = Counter(
        urlparse(item.get("url", "")).netloc for item in approved if item.get("url")
    )
    rejection_reasons = Counter(
        (item.get("reason") or "unspecified reason") for item in denied
    )
    return {
        "thinker": name,
        "approved_count": len(approved),
        "denied_count": len(denied),
        "top_domains": dict(top_domains.most_common()),
        "common_rejection_reasons": dict(rejection_reasons.most_common()),
    }

def main():
    print("SearchAI Goodrich - Resource Verification Agent")
    print("=" * 50)
    
    # Get person name from user
    name = input("Enter the name of the person you wish to research: ").strip()
    if not name:
        print("Error: Name cannot be empty")
        sys.exit(1)
    
    print(f"\nResearching: {name}")
    
    # Create output directory
    output_dir = create_output_directory(name)
    print(f"Output directory: {output_dir}")
    
    # Initialize clients
    openai_client = get_openai_client()
    claude_client = get_claude_client()
    
    prefilter_stats: Dict[str, int] = {}

    # Step 1: Get OpenAI sources
    print("\n1. Fetching sources from OpenAI...")
    openai_results = get_openai_sources(openai_client, name)

    # Save OpenAI results
    openai_file = output_dir / f"openai_{name}.json"
    save_json(openai_file, openai_results)
    print(f"OpenAI results saved to: {openai_file}")

    openai_results, openai_prefilter_removed = apply_prevalidation_filter(
        openai_results, "OpenAI"
    )
    prefilter_stats["openai"] = openai_prefilter_removed
    
    # Step 2: Get Claude sources
    print("\n2. Passing info to Claude...")
    try:
        claude_results = get_claude_sources(claude_client, name, openai_results)
    except RuntimeError as exc:
        print(f"⚠️ Claude request failed ({exc}). Using OpenAI fallback for additional sources.")
        claude_results = get_openai_additional_sources(openai_client, name, openai_results)

    # Save Claude results
    claude_file = output_dir / f"claude_{name}.json"
    save_json(claude_file, claude_results)
    print(f"Claude results saved to: {claude_file}")

    claude_results, claude_prefilter_removed = apply_prevalidation_filter(
        claude_results, "Claude"
    )
    prefilter_stats["claude"] = claude_prefilter_removed
    
    # Step 3: Validate all sources
    print("\n3. Validating all sources with evaluator...")
    validated_results = validate_sources(openai_client, name, openai_results, claude_results)

    # Save validated results
    validated_file = output_dir / f"validated_{name}.json"
    save_json(validated_file, validated_results)
    print(f"Validation results saved to: {validated_file}")

    # Generate manual review summary
    validation_report = generate_validation_report(
        name,
        validated_results.get("approved", []),
        validated_results.get("denied", []),
    )
    validation_report["pre_validation_removed"] = prefilter_stats
    report_file = output_dir / f"validation_report_{name}.json"
    save_json(report_file, validation_report)
    print(f"Validation report saved to: {report_file}")
    
    print(f"\n✅ Research complete for '{name}'")
    print(f"📁 All outputs saved in: {output_dir}")

if __name__ == "__main__":
    main()
