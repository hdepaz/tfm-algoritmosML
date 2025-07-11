import pandas as pd
import os

# Ruta del archivo original
input_dir = "/home/hdepaz/hdepaz/master/tfm2/UNSW-NB15/CSV-Files/entrenamiento/"
output_file = "/tmp/UNSW-datos_entrenamiento.csv"


# Mapeo protocolo IANA
proto_map = {
    "icmp": 1, "tcp": 6, "udp": 17,
    "gre": 47, "esp": 50, "ah": 51,
    "sctp": 132
}

# Campos finales seleccionados
selected_fields = [
    "sbytes", "sttl", "proto", "ct_dst_ltm", "is_sm_ips_ports",
    "Dintpkt", "ct_src_dport_ltm", "Dload", "Spkts", "Dpkts",
    "Smeansz", "Label"
]

dfs = []

# Procesar todos los CSVs
for filename in os.listdir(input_dir):
    if filename.endswith(".csv"):
        input_path = os.path.join(input_dir, filename)
        print(f"Procesando: {input_path}")

        # Leer y transformar
        df = pd.read_csv(input_path, low_memory=False)
        df_filtrado = df[selected_fields].copy()
        df_filtrado["proto"] = df_filtrado["proto"].str.lower().map(proto_map)
        df_filtrado = df_filtrado.dropna(subset=["proto"])
        df_filtrado["proto"] = df_filtrado["proto"].astype(int)

        dfs.append(df_filtrado)

# Concatenar todo y guardar en un solo CSV
df_final = pd.concat(dfs, ignore_index=True)
df_final["Label"] = df_final["Label"].map({1: "attack", 0: "normal"})
df_final.to_csv(output_file, index=False)
print(f"\n✅ CSV unificado generado en: {output_file}")
