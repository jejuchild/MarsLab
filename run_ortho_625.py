#!/usr/bin/env python3
"""Generate 6.25cm/px Mastcam-Z orthoimages for SR training."""
import sys, time, json
from pathlib import Path
from coregister.ortho import generate_ortho
from coregister.config import OUTPUT_DIR

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) > 1:
        sols = [int(s) for s in sys.argv[1:]]
    else:
        sols = []
        for sol_dir in sorted(OUTPUT_DIR.glob("sol?????")):
            if (sol_dir / "combined_lonlat.npz").exists():
                sol = int(sol_dir.name.replace("sol", ""))
                if not (OUTPUT_DIR / f"sol{sol:05d}_ortho.png").exists():
                    sols.append(sol)
    
    print(f"Generating {len(sols)} orthoimages @ 6.25cm/px")
    t0 = time.time()
    success = 0
    for i, sol in enumerate(sols, 1):
        print(f"\n[{i}/{len(sols)}] Sol {sol}")
        try:
            result = generate_ortho(sol, dtm=None, resolution_m=0.0625)
            if result:
                success += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
    
    print(f"\nDone: {success}/{len(sols)} in {(time.time()-t0)/60:.1f}min")

if __name__ == "__main__":
    main()
