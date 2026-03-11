import yaml
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# 1. 载入数据
yaml_data = """
edges:
- {cost: 1.84, from: node_00, to: node_01}
- {cost: 1.84, from: node_01, to: node_00}
- {cost: 2.79, from: node_00, to: node_02}
- {cost: 2.79, from: node_02, to: node_00}
- {cost: 1.71, from: node_00, to: node_04}
- {cost: 1.71, from: node_04, to: node_00}
- {cost: 1.83, from: node_00, to: node_05}
- {cost: 1.83, from: node_05, to: node_00}
- {cost: 1.86, from: node_01, to: node_02}
- {cost: 1.86, from: node_02, to: node_01}
- {cost: 2.32, from: node_01, to: node_04}
- {cost: 2.32, from: node_04, to: node_01}
- {cost: 1.88, from: node_02, to: node_03}
- {cost: 1.88, from: node_03, to: node_02}
- {cost: 1.86, from: node_02, to: node_04}
- {cost: 1.86, from: node_04, to: node_02}
- {cost: 2.36, from: node_03, to: node_04}
- {cost: 2.36, from: node_04, to: node_03}
- {cost: 2.34, from: node_04, to: node_05}
- {cost: 2.34, from: node_05, to: node_04}
nodes:
- {id: node_00, tags: [site1], x: 1.912, y: 1.396, yaw: 0.778}
- {id: node_01, tags: [site2], x: 3.199, y: 2.704, yaw: 1.602}
- {id: node_02, tags: [site3], x: 2.074, y: 4.184, yaw: 2.35}
- {id: node_03, tags: [site4], x: 0.42, y: 5.076, yaw: -1.577}
- {id: node_04, tags: [site5], x: 0.88, y: 2.757, yaw: -1.577}
- {id: node_05, tags: [site0], x: 0.326, y: 0.485, yaw: -0.006}
"""
data = yaml.safe_load(yaml_data)

# 2. 解析节点
node_dict = {}
tags_list = []
for node in data['nodes']:
    t = node['tags'][0] if node['tags'] else 'unknown'
    node_dict[node['id']] = {'x': node['x'], 'y': node['y'], 'yaw': node['yaw'], 'tag': t}
    if t not in tags_list: tags_list.append(t)
tags_list.sort()

# =====================================================
# 3. 自定义配置区 (论文微调核心)
# =====================================================
# 手动调整每个节点文字的位置: (x轴偏移, y轴偏移)
# 正数: 向右/向上 | 负数: 向左/向下
manual_offsets = {
    'node_00': (0.0, -0.35),  # site1 向下移动一点
    'node_01': (0.45, 0.0),   # site2 向右移动，避开线条
    'node_02': (0.45, 0.0),   # site3 向上移动
    'node_03': (0.0, 0.25),   # site4 向上移动
    'node_04': (-0.45, 0.0),  # site5 向左移动，避开中间密集的线
    'node_05': (0.0, -0.35),   # site0 向上移动，彻底避开坐标轴数字
}

TEXT_SIZE = 13        # 文字大小
NODE_SIZE = 350       # 节点圆圈大小
ARROW_LEN = 0.22      # 航向箭头长度
# =====================================================

# 4. 绘图风格设置
plt.rcParams['font.family'] = 'serif'
fig, ax = plt.subplots(figsize=(9, 10))
cmap = plt.get_cmap('Set2')
color_map = {tag: cmap(i) for i, tag in enumerate(tags_list)}

# 5. 绘制边 (Edges)
for edge in data['edges']:
    f_id, t_id = edge['from'], edge['to']
    if f_id in node_dict and t_id in node_dict:
        ax.plot([node_dict[f_id]['x'], node_dict[t_id]['x']], 
                [node_dict[f_id]['y'], node_dict[t_id]['y']], 
                color='gray', lw=1, alpha=0.3, zorder=1)

# 6. 绘制节点与微调后的文字
for n_id, info in node_dict.items():
    x, y, yaw, tag = info['x'], info['y'], info['yaw'], info['tag']
    color = color_map[tag]
    
    # 绘制节点圆圈
    ax.scatter(x, y, s=NODE_SIZE, color=color, edgecolors='white', linewidth=1.5, zorder=5)
    
    # 绘制航向箭头
    ax.arrow(x, y, ARROW_LEN * np.cos(yaw), ARROW_LEN * np.sin(yaw), 
             width=0.02, head_width=0.1, head_length=0.12, fc='k', ec='k', zorder=6)
    
    # 获取手动偏移量
    dx, dy = manual_offsets.get(n_id, (0.0, -0.3))
    
    # 绘制文字 (ID + Tag)
    ax.text(x + dx, y + dy, f"{n_id}\n({tag})", 
            ha='center', va='center',  # 居中对齐，完全靠 dx, dy 控制
            fontsize=TEXT_SIZE, 
            fontweight='bold', 
            linespacing=1.2, 
            family='sans-serif', 
            zorder=7)

# 7. 坐标轴与边界优化
ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.3, zorder=0)

# 增加绘图余量，防止边缘节点显示不全
all_x = [n['x'] for n in node_dict.values()]
all_y = [n['y'] for n in node_dict.values()]
ax.set_xlim(min(all_x) - 0.6, max(all_x) + 0.8)
ax.set_ylim(min(all_y) - 0.6, max(all_y) + 0.6)

ax.set_xlabel('Global X (m)', fontsize=11)
ax.set_ylabel('Global Y (m)', fontsize=11)
ax.set_title('Topological Semantic Map of the Environment', pad=20, fontsize=13)

# 8. 图例
legend_elements = [Patch(facecolor=color_map[t], edgecolor='white', label=t) for t in tags_list]
ax.legend(handles=legend_elements, title="Semantic Labels", loc='upper left', 
          bbox_to_anchor=(1.02, 1),title_fontsize=13, frameon=True,fontsize=13, shadow=False)

# 9. 保存矢量图和高清图
plt.tight_layout()
plt.savefig('topo_map_final.pdf', bbox_inches='tight')
plt.savefig('topo_map_final.png', dpi=300, bbox_inches='tight')
print("Successfully generated: 'topo_map_final.pdf' (Vector) and 'topo_map_final.png'")
plt.show()
