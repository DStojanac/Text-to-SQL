# Thesis Text-to-SQL Project

## Overview

This project implements a schema-aware and secure natural-language-to-SQL system for a master's thesis.

## Main Goals

- convert natural-language questions into SQL queries
- fine-tune a pretrained sequence-to-sequence model
- validate generated SQL before execution
- execute only safe read-only queries
- provide a backend that can later connect to a frontend application

## Planned Components

- dataset preprocessing
- schema serialization
- baseline inference
- fine-tuning
- SQL validation
- execution-guided repair
- API backend
- frontend UI
