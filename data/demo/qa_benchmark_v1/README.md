# QA Benchmark v1

This benchmark package is designed for the current small-document product demo.

Contents:

- `docs/`: 5 domain-specific markdown documents
- `qa_dataset.json`: 50 QA pairs with ground-truth answers

Design rules:

- 5 domains
- 10 QA pairs per domain
- each QA includes:
  - `qa_id`
  - `domain_id`
  - `document_file`
  - `expected_expert`
  - `query`
  - `answer`

The 5 domains are:

1. Insurance
2. Banking
3. Healthcare
4. Retail
5. Industrial Equipment
