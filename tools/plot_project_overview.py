import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def draw_box(ax, xy, w, h, text, fontsize=10, facecolor='#f0f0f0'):
    rect = patches.FancyBboxPatch(xy, w, h, boxstyle='round,pad=0.3',
                                  linewidth=1, edgecolor='k', facecolor=facecolor)
    ax.add_patch(rect)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha='center', va='center', fontsize=fontsize)


def render_overview(out_path='outputs/project_overview.png'):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axis('off')

    # Title
    ax.text(0.5, 0.95, 'Project Overview — DeepRL Group Cooperation', ha='center', va='center', fontsize=16, weight='bold')

    # src box
    src_text = '''src/
  - environment/ (env, payoffs)
  - agents/ (human bots, llm bots)
  - planner/ (GNN model, policy)
  - training/ (trainer, updates)
  - utils/ (metrics, viz)'''
    draw_box(ax, (0.05, 0.55), 0.45, 0.35, src_text, fontsize=10)

    # Scripts box
    scripts_text = '''Top-level scripts
  - train.py (training loop)
  - test_model.py (eval + plots)
  - 100episode.py (batch eval)'''
    draw_box(ax, (0.55, 0.55), 0.4, 0.18, scripts_text, fontsize=10)

    # Configs / models
    configs_text = '''configs/ (yaml)
  - saved_models/ (.pth snapshots)
  - README.md / requirements.txt'''
    draw_box(ax, (0.55, 0.3), 0.4, 0.18, configs_text, fontsize=10)

    # Flow arrows
    ax.annotate('', xy=(0.35, 0.7), xytext=(0.55, 0.7), arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.annotate('', xy=(0.75, 0.48), xytext=(0.75, 0.4), arrowprops=dict(arrowstyle='->', lw=1.5))

    # Footer notes
    ax.text(0.5, 0.08, 'High-level: Planner (GNN) suggests edge changes → Bots accept/reject → Env updates → Rewards computed → Trainer updates planner',
            ha='center', va='center', fontsize=10)

    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved overview to {out_path}")


if __name__ == '__main__':
    render_overview()
