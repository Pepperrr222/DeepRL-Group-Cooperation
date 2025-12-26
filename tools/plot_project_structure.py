import os
import textwrap
import matplotlib.pyplot as plt


def tree_lines(root='.', max_depth=4, prefix=''):
    lines = []
    entries = []
    try:
        entries = sorted(os.listdir(root))
    except Exception:
        return lines

    for i, name in enumerate(entries):
        path = os.path.join(root, name)
        connector = '└── ' if i == len(entries) - 1 else '├── '
        lines.append(prefix + connector + name)
        if os.path.isdir(path) and (prefix.count('│') // 1) < max_depth:
            extension = '    ' if i == len(entries) - 1 else '│   '
            lines.extend(tree_lines(path, max_depth=max_depth, prefix=prefix + extension))
    return lines


def render_tree_image(root='.', out_path='outputs/project_structure.png', max_depth=4):
    lines = [root.rstrip('/')]
    lines += tree_lines(root, max_depth=max_depth, prefix='')
    text = '\n'.join(lines)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig = plt.figure(figsize=(8, max(4, len(lines) * 0.2)))
    plt.axis('off')
    plt.text(0, 1, text, fontfamily='monospace', fontsize=10, va='top')
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved project structure image to {out_path}")


if __name__ == '__main__':
    render_tree_image(root='.', out_path='outputs/project_structure.png', max_depth=4)
