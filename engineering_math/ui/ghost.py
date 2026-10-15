"""같은 구성을 다시 계산했을 때 이전 결과의 곡선을 흐린 파선으로 겹쳐 보여 줍니다.

그래프를 만든 직후 실선의 데이터를 저장해 두었다가(snapshot), 다음 결과의 같은 이름·같은 위치의
축에 겹쳐 그립니다(overlay). 파선이므로 재생 막대는 이 선을 기준선으로 보고 움직이지 않습니다.
색 지도·등고선은 겹쳐 그리면 알아볼 수 없으므로 선 그래프만 대상으로 합니다.
"""
import numpy as np
from matplotlib.lines import Line2D

GHOST_LABEL = '_이전 결과'
OWN_LABELS = (GHOST_LABEL, '_재생 점')
GHOST_STYLE = dict(linestyle='--', linewidth=1.2, alpha=.4, zorder=1)
MAX_DIFF_ITEMS = 3


def _axes(figure):
    return [axes for axes in figure.axes if axes.get_label() != '<colorbar>']


def snapshot(figures):
    """{그래프 이름: [축마다 [(x, y, 색), …]]}. 그래프를 재생하거나 확대하기 전에 불러야 합니다."""
    result = {}
    for name, figure in figures.items():
        result[name] = [[(np.array(line.get_xdata(), float), np.array(line.get_ydata(), float), line.get_color())
                         for line in axes.lines
                         if line.get_linestyle() in ('-', 'solid') and np.ndim(line.get_xdata()) and len(line.get_xdata()) > 2
                         and line.get_label() not in OWN_LABELS]
                        for axes in _axes(figure)]
    return result


def describe(previous, current, specs):
    """이전 결과와 달라진 입력을 'N=10, alpha=0.1'처럼 짧게 적습니다."""
    changed = [spec.key for spec in specs if previous.get(spec.key) != current.get(spec.key)]
    parts = []
    for key in changed[:MAX_DIFF_ITEMS]:
        value = previous.get(key)
        text = f'{value:g}' if isinstance(value, float) else str(value)
        parts.append(f'{key}={text if len(text) <= 18 else text[:16] + "…"}')
    if len(changed) > MAX_DIFF_ITEMS:
        parts.append('…')
    return ', '.join(parts)


def overlay(figures, previous, label):
    """previous(snapshot 결과)의 곡선을 figures의 같은 축에 겹칩니다. 겹친 선의 수를 돌려줍니다."""
    count = 0
    for name, figure in figures.items():
        old_axes = previous.get(name)
        new_axes = _axes(figure)
        if old_axes is None or len(old_axes) != len(new_axes):
            continue
        for axes, lines in zip(new_axes, old_axes):
            if not lines or not any(line.get_linestyle() in ('-', 'solid') for line in axes.lines):
                continue
            limits = axes.get_xlim(), axes.get_ylim()
            for x, y, color in lines:
                axes.plot(x, y, color=color, label=GHOST_LABEL, **GHOST_STYLE)
                count += 1
            # 이전 곡선이 범위를 벗어나도 현재 결과의 보기 범위는 유지합니다.
            axes.set_xlim(*limits[0])
            axes.set_ylim(*limits[1])
            legend = axes.get_legend()
            if legend is not None:
                handles, labels = axes.get_legend_handles_labels()
                proxy = Line2D([], [], color='gray', **{**GHOST_STYLE, 'alpha': .7})
                axes.legend(handles+[proxy], labels+[f'Previous ({label})' if label else 'Previous'],
                            loc=legend._loc, fontsize=legend.prop.get_size())
    return count
