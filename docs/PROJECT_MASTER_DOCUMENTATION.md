# Combat Commentary Assistant
## Project Master Documentation

Version: 1.0  
Status: Architecture Redesign


# 1. Project Overview

Combat Commentary Assistant adalah sistem AI berbasis video
yang dirancang untuk menghasilkan structured commentary script
pada pertandingan olahraga tarung secara otomatis.

Sistem tidak hanya menghasilkan deskripsi kejadian,
tetapi memahami konteks pertandingan, identitas fighter,
karakter bertarung, serta membangun narasi berdasarkan
Fighter Narrative DNA.


# 2. Core Concept

Input utama sistem:

VIDEO PERTANDINGAN


User hanya memberikan video sebagai input utama.

Tidak diperlukan input manual dari user seperti:

- nama fighter
- event
- timestamp
- momen pertandingan


Sistem bertugas mengekstraksi informasi tersebut dari video
melalui proses analisis AI.


# 3. Final Output

Output sistem berupa structured commentary script yang
dihasilkan setelah proses analisis video selesai.

Output hanya dapat diakses oleh pengguna yang memiliki
otorisasi sesuai role sistem.


Format:


MATCH:
[Fighter A vs Fighter B]


EVENT:
[Competition/Event]


------------------------------------------------


[TIMESTAMP]


MOMENT:
[Deskripsi kejadian]


ANALYSIS:
[Makna kejadian terhadap pertandingan]


COMMENTARY:

[Narasi dengan gaya storytelling]


------------------------------------------------



# 4. Product Philosophy

Sistem tidak dibuat sebagai AI komentar generik.

Tujuan utama adalah membuat AI yang mampu memahami:

- siapa fighter tersebut
- bagaimana perjalanan fighter
- karakter bertarung
- konflik pertandingan
- perubahan momentum


Output harus terasa seperti narasi komentator profesional
yang memiliki konteks dan identitas berbeda untuk setiap fighter.


# 5. Developer Controlled Knowledge Base

Salah satu pembeda utama sistem adalah adanya
knowledge base internal yang hanya dikontrol developer.

User hanya memberikan video.

User tidak dapat:

- mengubah karakter AI
- memberikan style commentary sendiri
- memasukkan database fighter


Knowledge base menjadi sumber pemahaman AI.


# 5.1 User Access Control

Sistem menerapkan role-based access control untuk menjaga
keamanan aset AI dan knowledge base internal.


## Operator

Operator memiliki akses untuk:

- mengunggah video pertandingan
- melihat status proses analisis
- mengambil hasil generated script


## Admin / Developer

Admin memiliki akses untuk:

- mengelola Fighter Narrative DNA
- memperbarui knowledge base
- melakukan konfigurasi sistem


Knowledge base, model configuration, dan logika internal AI
tidak dapat diakses oleh user umum.


# 6. Fighter Narrative DNA

Knowledge base fighter tidak berisi template komentar.

Data yang disimpan:


## Fighter Background

Riwayat dan perjalanan fighter.


## Fight Identity

Cara bertarung:

- pressure fighter
- counter striker
- technical fighter


## Personality

Karakter:

- agresif
- tenang
- calculated
- relentless


## Narrative Theme

Tema cerita:

- pembuktian
- comeback
- veteran experience
- rising star


## Vocabulary DNA

Kata dan gaya bahasa yang sesuai
dengan karakter fighter.


Contoh:


Fighter agresif:

- tekanan
- mendominasi
- tidak memberi ruang


Fighter counter:

- membaca celah
- menunggu momentum
- menghukum kesalahan



# 7. System Architecture


Operator / Authorized User

↓

Frontend Application

↓

Spring Boot Backend

↓

AI Processing Service

↓

Video Understanding

↓

Fight Event Extraction

↓

Fighter Narrative DNA Integration

↓

Narrative Generation Engine

↓

Structured Commentary Script Output



Admin / Developer

↓

Knowledge Base Management

↓

Fighter Narrative DNA



# 8. AI Pipeline


## Stage 0
Video Input

User hanya memberikan video pertandingan sebagai input utama.

Sistem tidak bergantung pada metadata manual
karena informasi pertandingan akan diekstraksi
dari video.


---

## Stage 1
Video Processing

Tujuan:

Memahami isi video.


Output:

- fighter detection
- movement detection
- action recognition


---

## Stage 2
Fight Event Detection

Mencari kejadian penting:

- pressure
- counter
- significant strike
- knockdown
- momentum shift


Output:

timestamp events


---

## Stage 3
Fighter Context Injection

Menggabungkan:

Video analysis

+

Fighter Narrative DNA


Tahap ini membuat narasi memiliki karakter
yang berbeda untuk setiap fighter.


---

## Stage 4
Narrative Generation

Menghasilkan commentary script
dengan gaya bahasa dan konteks yang konsisten.



# 9. Technology Direction


Backend:

Java Spring Boot


AI Service:

Python


Database:

PostgreSQL


Container:

Docker


Future AI Components:

- Computer Vision Model
- Speech Recognition
- Large Language Model



# 10. Development Roadmap


## Phase 1
Backend Foundation

Status:
DONE


---

## Phase 2
Architecture Alignment

Status:
IN PROGRESS


Scope:

- finalize system architecture
- define user roles
- define AI pipeline
- define knowledge base structure


---

## Phase 3
Video Upload System

Status:
NEXT


---

## Phase 4
AI Processing Service

Status:
PLANNED


---

## Phase 5
Knowledge Base Engine

Status:
PLANNED


---

## Phase 6
Narrative Generation

Status:
PLANNED


---

## Phase 7
Frontend Dashboard

Status:
PLANNED



# 11. Engineering Principle


Project dikembangkan dengan pendekatan enterprise:

- modular architecture
- scalable service
- documented decision
- clean separation between backend, AI, and knowledge layer


Dokumen ini menjadi referensi utama pengembangan.