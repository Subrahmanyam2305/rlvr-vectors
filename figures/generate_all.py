#!/usr/bin/env python3
"""Generate all paper figures."""
from fig1_schematic import draw as fig1
from fig2_spectral import draw as fig2
from fig3_gate_scatter import draw as fig3
from fig4_5_forest import draw as fig4_5
from appendix_figs import fig_a1_heatmap, fig_a2_gate_by_block, fig_a3_val_curves, fig_a4_protocol

if __name__ == "__main__":
    print("=== Main Figures ===")
    fig1()
    fig2()
    fig3()
    fig4_5()
    print("\n=== Appendix Figures ===")
    fig_a1_heatmap()
    fig_a2_gate_by_block()
    fig_a3_val_curves()
    fig_a4_protocol()
    print("\nDone.")
