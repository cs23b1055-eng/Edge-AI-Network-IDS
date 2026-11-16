"""Packet sources: live sniffing (needs CAP_NET_RAW/root) or offline PCAP replay. Both yield PacketRecord."""
import threading, time
from dataclasses import dataclass
import scapy.layers.l2, scapy.layers.inet  # noqa: F401  (registers link-type -> layer mappings for PcapReader)
from typing import Iterator, Optional

@dataclass(slots=True)
class PacketRecord:
    ts: float          # capture timestamp (seconds)
    src: str
    dst: str
    sport: int
    dport: int
    proto: int         # 6 = TCP, 17 = UDP
    size: int          # IP total length (bytes on the wire at L3)
    payload: int       # transport payload bytes
    hdr_len: int       # transport header bytes (TCP data offset*4, UDP 8)
    flags: int         # TCP flag bits (0 for UDP)
    window: int        # TCP window (0 for UDP)

FIN, SYN, RST, PSH, ACK, URG, ECE, CWR = 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80

def parse_packet(pkt) -> Optional[PacketRecord]:
    """Convert a scapy packet into a PacketRecord. Non-IPv4 TCP/UDP packets return None."""
    from scapy.layers.inet import IP, TCP, UDP
    if not pkt.haslayer(IP):
        return None
    ip = pkt[IP]
    if ip.frag and ip.frag > 0:      # non-first fragment carries no transport header
        return None
    if ip.haslayer(TCP):
        t = ip[TCP]; hdr = (t.dataofs or 5) * 4
        payload = max(0, ip.len - ip.ihl * 4 - hdr)
        return PacketRecord(float(pkt.time), ip.src, ip.dst, int(t.sport), int(t.dport), 6, int(ip.len), payload, hdr, int(t.flags), int(t.window))
    if ip.haslayer(UDP):
        u = ip[UDP]; payload = max(0, int(u.len) - 8)
        return PacketRecord(float(pkt.time), ip.src, ip.dst, int(u.sport), int(u.dport), 17, int(ip.len), payload, 8, 0, 0)
    return None

class CaptureStatus:
    def __init__(self):
        self.state = "idle"; self.total_seen = 0; self.parsed = 0; self.ignored = 0; self.error = ""

def pcap_source(path, status: CaptureStatus, speed: float = 0.0, stop: Optional[threading.Event] = None) -> Iterator[PacketRecord]:
    """Replay a PCAP. speed=0 -> as fast as possible; speed=1 -> original timing; N -> Nx faster."""
    from scapy.utils import PcapReader
    status.state = "replaying"; first_ts = first_wall = None
    try:
        with PcapReader(path) as rd:
            for pkt in rd:
                if stop is not None and stop.is_set(): break
                status.total_seen += 1
                rec = parse_packet(pkt)
                if rec is None: status.ignored += 1; continue
                if speed > 0:
                    if first_ts is None: first_ts, first_wall = rec.ts, time.time()
                    delay = (rec.ts - first_ts) / speed - (time.time() - first_wall)
                    if delay > 0: time.sleep(delay)
                status.parsed += 1
                yield rec
        status.state = "finished"
    except Exception as e:
        status.state = "error"; status.error = str(e); raise

def live_source(iface: Optional[str], status: CaptureStatus, bpf: str = "ip and (tcp or udp)", stop: Optional[threading.Event] = None) -> Iterator[PacketRecord]:
    """Live capture with scapy (AF_PACKET). Requires root or `setcap cap_net_raw+eip` on the python binary."""
    import queue
    from scapy.all import sniff
    q: "queue.Queue" = queue.Queue(maxsize=50000); stop = stop or threading.Event()
    def cb(pkt):
        status.total_seen += 1
        rec = parse_packet(pkt)
        if rec is None: status.ignored += 1; return
        try: q.put_nowait(rec)
        except queue.Full: status.ignored += 1
    def run():
        try:
            status.state = "capturing"
            sniff(iface=iface, prn=cb, store=False, filter=None, stop_filter=lambda p: stop.is_set())
        except Exception as e:
            status.state = "error"; status.error = f"{type(e).__name__}: {e}"; stop.set()
    th = threading.Thread(target=run, daemon=True); th.start()
    while not stop.is_set() or not q.empty():
        try: rec = q.get(timeout=0.3)
        except queue.Empty: continue
        status.parsed += 1; yield rec
    if status.state != "error": status.state = "stopped"
