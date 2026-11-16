"""Bidirectional flow assembly keyed by the 5-tuple, with FIN/RST + idle + active timeouts."""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from packet_capture import PacketRecord, FIN, RST

@dataclass
class Flow:
    key: tuple
    src: str; dst: str; sport: int; dport: int; proto: int     # orientation of the FIRST packet (= "forward")
    start: float; last: float
    fwd: List[PacketRecord] = field(default_factory=list)
    bwd: List[PacketRecord] = field(default_factory=list)
    order: List[Tuple[float, int]] = field(default_factory=list)   # (ts, size) of all packets in arrival order
    fin_fwd: bool = False; fin_bwd: bool = False; rst: bool = False

    @property
    def n_packets(self): return len(self.fwd) + len(self.bwd)

def flow_key(p: PacketRecord):
    a, b = (p.src, p.sport), (p.dst, p.dport)
    return (p.proto,) + ((a, b) if a <= b else (b, a))

class FlowManager:
    def __init__(self, idle_timeout=30.0, active_timeout=120.0, max_flows=100000):
        self.idle_timeout, self.active_timeout, self.max_flows = idle_timeout, active_timeout, max_flows
        self.close_grace = 2.0   # seconds to wait for the final ACK after both FINs
        self.flows: Dict[tuple, Flow] = {}
        self.total_flows_created = 0

    def add(self, p: PacketRecord) -> List[Flow]:
        """Add a packet; returns flows that completed because of it (FIN-FIN / RST)."""
        done: List[Flow] = []
        k = flow_key(p); f = self.flows.get(k)
        if f is not None and (p.ts - f.last > self.idle_timeout or p.ts - f.start > self.active_timeout):
            done.append(self.flows.pop(k)); f = None
        if f is None:
            if len(self.flows) >= self.max_flows:        # memory guard: evict the oldest flow
                oldest = min(self.flows.values(), key=lambda x: x.last); done.append(self.flows.pop(oldest.key))
            f = Flow(k, p.src, p.dst, p.sport, p.dport, p.proto, p.ts, p.ts); self.flows[k] = f; self.total_flows_created += 1
        fwd = (p.src, p.sport) == (f.src, f.sport)
        was_closing = f.fin_fwd and f.fin_bwd      # teardown already complete: this packet is the final ACK
        (f.fwd if fwd else f.bwd).append(p); f.order.append((p.ts, p.size)); f.last = max(f.last, p.ts)
        if p.proto == 6:
            if p.flags & RST: f.rst = True
            if p.flags & FIN:
                if fwd: f.fin_fwd = True
                else: f.fin_bwd = True
        if f.rst or was_closing:
            done.append(self.flows.pop(k))
        return done

    def expire(self, now: float) -> List[Flow]:
        out = [f for f in self.flows.values() if now - f.last > self.idle_timeout or now - f.start > self.active_timeout
               or (f.fin_fwd and f.fin_bwd and now - f.last > self.close_grace)]
        for f in out: del self.flows[f.key]
        return out

    def flush(self) -> List[Flow]:
        out = list(self.flows.values()); self.flows.clear(); return out
