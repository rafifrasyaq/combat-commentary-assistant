# Combat Commentary Assistant Architecture

## Overview

Combat Commentary Assistant adalah sistem AI berbasis video yang menghasilkan
structured commentary script dari pertandingan olahraga tarung.

Sistem menggunakan pendekatan modular enterprise dengan pemisahan antara:

- Frontend layer
- Backend orchestration layer
- AI processing layer
- Knowledge base layer
- Data infrastructure


## High Level Architecture


User
 |
 |
Frontend Application
 |
 |
Spring Boot Backend
 |
 |---------------------
 |                    |
 |                    |
AI Processing      Database
Service            PostgreSQL
 |
 |
Video Understanding
 |
 |
Fight Event Detection
 |
 |
Fighter Narrative Context
 |
 |
Narrative Generation
 |
 |
Commentary Script Output



## Component Description


## Frontend

Bertanggung jawab untuk:

- authentication
- video submission
- monitoring analysis process
- accessing authorized results


Frontend memiliki pembagian akses:

### Operator

Akses:

- upload video pertandingan
- melihat status proses
- mengambil hasil analisis


### Admin / Developer

Akses:

- mengelola fighter knowledge base
- mengelola narrative DNA
- monitoring sistem


---

## Backend Service

Spring Boot berfungsi sebagai orchestration layer.

Tanggung jawab:

- authentication
- request management
- video processing workflow
- komunikasi dengan AI service
- penyimpanan hasil


---

## AI Processing Service

Service Python menangani:

- video processing
- computer vision
- audio processing
- fight event extraction
- narrative generation


---

## Knowledge Base Layer

Knowledge base merupakan aset internal developer.

Berisi:

- fighter background
- fighting identity
- personality
- narrative theme
- vocabulary DNA


Knowledge base tidak dapat dimodifikasi oleh user umum.


---

## Output System

Output berupa structured commentary script:

- Match
- Event
- Timestamp
- Moment
- Analysis
- Commentary


Tujuan sistem bukan membuat komentar generik,
tetapi menghasilkan narasi yang mempertimbangkan
identitas dan cerita fighter.