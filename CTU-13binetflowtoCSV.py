import csv
import glob
from datetime import datetime

# Carpeta con archivos .binetflow (puedes cambiar el path)
input_folder = "/home/hdepaz/hdepaz/master/tfm2/fuentes/test-CTU13/"
output_file = "input_test_bueno.csv"

selected_fields = [
    "StartTime", "Dur", "Proto", "SrcAddr", "Sport", "DstAddr", "Dport",
    "TotPkts", "TotBytes", "SrcBytes", "Label"
]

proto_map = {
    "icmp": 1,
    "tcp": 6,
    "udp": 17
}

def clean_port(port):
    port = port.strip()
    if port.startswith("0x") or not port.isdigit():
        return "0"  #Si está en hexadecimal o no es numérico, lo dejamos en "0"
    return port

def to_epoch_micro(ts):
    dt = datetime.strptime(ts, "%Y/%m/%d %H:%M:%S.%f")
    return dt.timestamp()

def get_label(flow_str):
    if "From-Botnet" in flow_str:
        return "Botnet"
    elif "From-Normal" in flow_str or "To-Normal" in flow_str:
        return "Normal"
    else:
        return "Background"

# Abrimos el CSV de salida una vez, escribimos encabezado y vamos añadiendo filas
with open(output_file, "w", newline="") as outfile:
    writer = csv.DictWriter(outfile, fieldnames=selected_fields)
    writer.writeheader()

    # Iterar todos los archivos binnetflow en la carpeta
    for filename in glob.glob(input_folder + "*.binetflow"):
        print ("Processing: ",filename)
        with open(filename, "r") as infile:
            for line in infile:
                line = line.strip()
                if not line or line.startswith("StartTime"):
                    continue

                parts = [p.strip() for p in line.split(",")]

                try:
                    proto_text = parts[2].lower()
                    proto_num = proto_map.get(proto_text, -1)
                    sport = clean_port(parts[4])
                    dport = clean_port(parts[7])
                    flow_field = parts[-1]
                    src_addr = parts[3]
                    dst_addr = parts[6]

                    if ':' in src_addr or ':' in dst_addr:
                        print ("ERROR DE IP ", src_addr, " " ,dst_addr)
                        continue

                    row = {
                        "StartTime": to_epoch_micro(parts[0]),
                        "Dur": parts[1],
                        "Proto": proto_num,
                        "SrcAddr": parts[3],
                        "Sport": sport,
                        "DstAddr": parts[6],
                        "Dport": dport,
                        "TotPkts": parts[11],
                        "TotBytes": parts[12],
                        "SrcBytes": parts[13],
                        "Label": get_label(flow_field)
                    }

                    writer.writerow(row)

                except Exception as e:
                    print(f"Error procesando línea en {filename}:\n{line}\n{e}")

