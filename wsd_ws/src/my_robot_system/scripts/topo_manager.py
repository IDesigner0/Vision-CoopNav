# scripts/topo_manager.py
import yaml
import heapq
from collections import defaultdict
import os

class TopoMap:
    def __init__(self, yaml_path=None):
        if yaml_path is None:
            # 默认从 config 目录加载
            yaml_path = os.path.join(os.path.dirname(__file__), "../config/topo_map.yaml")
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        self.nodes = {}
        for n in data['nodes']:
            self.nodes[n['id']] = {
                'x': n['x'],
                'y': n['y'],
                'yaw': n['yaw'],
                'tag': n['tags'][0] if n['tags'] else None
            }
        
        self.graph = defaultdict(list)
        for e in data['edges']:
            self.graph[e['from']].append((e['to'], e['cost']))
            self.graph[e['to']].append((e['from'], e['cost']))

    def find_node_by_tag(self, tag):
        for node_id, info in self.nodes.items():
            if info['tag'] == tag:
                return node_id
        return None

    def dijkstra(self, start, goal):
        dist = {node: float('inf') for node in self.nodes}
        prev = {node: None for node in self.nodes}
        dist[start] = 0
        pq = [(0, start)]

        while pq:
            d, u = heapq.heappop(pq)
            if u == goal:
                break
            for v, cost in self.graph[u]:
                alt = d + cost
                if alt < dist[v]:
                    dist[v] = alt
                    prev[v] = u
                    heapq.heappush(pq, (alt, v))
        
        path = []
        curr = goal
        while curr:
            path.append(curr)
            curr = prev[curr]
        return path[::-1] if path and path[-1] == start else []
