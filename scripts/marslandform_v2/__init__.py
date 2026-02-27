# MarsLandformNet V2 - Agentic Mars Landform Classification
"""
Agentic Mars landform classification system using:
- SSL-adapted DINOv2 ViT-B/14 + LoRA
- Attention-based Multiple Instance Learning (MIL)
- RAG-augmented domain knowledge (SPECTER2 + ChromaDB)
- ReACT-loop VLM reasoning agent

Classes: LDA (Lobate Debris Apron), LVF (Lineated Valley Fill),
         CCF (Concentric Crater Fill), GLF (Glacier-Like Form)
"""
__version__ = "2.0.0"
