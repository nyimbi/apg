# Accounts Payable Feature Guide

## Vendor Controls

Vendor records require owner, tax profile, and payment method evidence. Bank-change records require review before acceptance.

## Invoice Controls

Invoices require a registered vendor, invoice number, currency, positive amount, and document reference. Potential duplicates are routed for review.

## Matching Controls

PO-backed invoices require receipt evidence. Variance above the configured threshold requires a variance review before matching can proceed.

## Approval Controls

High-value invoices require approval evidence, and requesters cannot approve their own invoices.

## Payment Controls

Payments require an approved, unheld invoice, positive payment amount, and cash account. Payment batches require review before release.

## Expense Controls

Expense reports require employee identity and receipt evidence. Policy exceptions require review.

## Close Controls

Period close is blocked by open AP exceptions, unposted invoices, or missing aging review.

## Agent Controls

AP agents are first-class capability citizens with explicit runtime and role support. Privileged AP-agent actions require recorded human approval.
