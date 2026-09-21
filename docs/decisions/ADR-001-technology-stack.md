# ADR-001: Technology Stack Selection

## Status

Accepted

## Context

Combat Commentary Assistant requires a scalable architecture combining enterprise backend development and AI processing capability.

## Decision

The system will use:

Backend:
- Java Spring Boot

AI Service:
- Python FastAPI

Frontend:
- React

Database:
- PostgreSQL

Infrastructure:
- Docker

## Rationale

Java Spring Boot is selected for backend services because of its mature enterprise ecosystem, strong security support, and scalability.

Python FastAPI is selected for AI services because of its extensive machine learning and natural language processing ecosystem.

PostgreSQL is selected because the application requires structured relational data management.

Docker is selected to provide consistent development and deployment environments.