"""Pathfinder — AI-powered Mars rover route planning engine.

Inspired by NASA JPL's Perseverance AI-planned drive system (Dec 2025).
Implements the ground-based cognitive layer: cost map generation,
Field D* path planning, waypoint generation, and route analysis.

References:
    [1] NASA JPL, "Perseverance Rover Completes First AI-Planned Drive," 2026
    [2] Ferguson & Stentz, "Field D*," J. Field Robotics, 2006
    [3] Carsten et al., "Global Path Planning on MER," IEEE Aerospace, 2007
    [4] Anthropic, "Claude AI Powers NASA's First AI-Planned Mars Rover Drive"
"""
