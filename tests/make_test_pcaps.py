"""Builds SYNTHETIC packet fixtures with scapy (for flow/feature unit tests and the offline demo).
These are NOT training data and NOT real attack captures; model predictions on them are only a smoke test."""
import os, random, sys
from scapy.all import IP, TCP, UDP, Raw, wrpcap, Ether
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "datasets", "pcaps")
os.makedirs(OUT, exist_ok=True)

def pkt(t, p):
    p = Ether() / p; p.time = t; return p

def tcp_session(t0, c, s, cport, sport, req=b"GET / HTTP/1.1\r\nHost: x\r\n\r\n", resp=b"HTTP/1.1 200 OK\r\n\r\n" + b"a" * 1200, gap=0.01):
    t = t0; out = []
    def add(src, dst, sp, dp, flags, data=b"", seq=0, ack=0, win=29200):
        nonlocal t
        out.append(pkt(t, IP(src=src, dst=dst) / TCP(sport=sp, dport=dp, flags=flags, seq=seq, ack=ack, window=win) / (Raw(data) if data else b""))); t += gap
    add(c, s, cport, sport, "S", win=29200); add(s, c, sport, cport, "SA", win=14600); add(c, s, cport, sport, "A")
    add(c, s, cport, sport, "PA", req); add(s, c, sport, cport, "PA", resp); add(c, s, cport, sport, "A")
    add(c, s, cport, sport, "FA"); add(s, c, sport, cport, "FA"); add(c, s, cport, sport, "A")
    return out

def main():
    random.seed(1)
    normal = []
    for i in range(30): normal += tcp_session(1000 + i * 0.5, "192.168.1.10", "93.184.216.34", 40000 + i, 80)
    for i in range(10):   # DNS
        t = 1000 + i * 0.7
        normal += [pkt(t, IP(src="192.168.1.10", dst="8.8.8.8") / UDP(sport=50000 + i, dport=53) / Raw(b"q" * 30)),
                   pkt(t + 0.02, IP(src="8.8.8.8", dst="192.168.1.10") / UDP(sport=53, dport=50000 + i) / Raw(b"r" * 80))]
    scan = []   # SYN scan: one source probes many ports; closed ports answer RST
    for i, port in enumerate(range(1, 400)):
        t = 2000 + i * 0.002
        scan.append(pkt(t, IP(src="10.0.0.99", dst="192.168.1.20") / TCP(sport=45000, dport=port, flags="S", window=1024)))
        scan.append(pkt(t + 0.0005, IP(src="192.168.1.20", dst="10.0.0.99") / TCP(sport=port, dport=45000, flags="RA", window=0)))
    flood = []  # many short high-rate HTTP GET-style flows from varied source ports
    for i in range(300):
        t = 3000 + i * 0.003
        flood += [pkt(t, IP(src="10.0.0.50", dst="192.168.1.20") / TCP(sport=30000 + i, dport=80, flags="S", window=8192)),
                  pkt(t + 0.0004, IP(src="10.0.0.50", dst="192.168.1.20") / TCP(sport=30000 + i, dport=80, flags="PA", window=8192) / Raw(b"GET /?" + b"x" * 40))]
    for name, pk in {"synthetic_normal.pcap": normal, "synthetic_portscan.pcap": scan, "synthetic_flood.pcap": flood}.items():
        wrpcap(os.path.join(OUT, name), sorted(pk, key=lambda p: p.time)); print(name, len(pk), "packets")
    wrpcap(os.path.join(OUT, "synthetic_mixed.pcap"), sorted(normal + scan + flood, key=lambda p: p.time))
main()
