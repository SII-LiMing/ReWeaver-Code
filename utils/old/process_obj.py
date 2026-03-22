
def strip_vt_from_obj(input_path, output_path):
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            # 跳过纹理坐标行
            if line.startswith("vt "):
                continue
            # 修改 f 行，删除 vt 索引
            elif line.startswith("f "):
                parts = line.strip().split()
                new_faces = []
                for part in parts[1:]:
                    if '/' in part:
                        v_idx = part.split('/')[0]
                        new_faces.append(v_idx)
                    else:
                        new_faces.append(part)
                outfile.write("f " + " ".join(new_faces) + "\n")
            else:
                outfile.write(line)


def strip_vn_from_obj(input_path, output_path):
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            # 跳过纹理坐标行
            if line.startswith("vn "):
                continue
            # 修改 f 行，删除 vt 索引
            elif line.startswith("f "):
                parts = line.strip().split()
                new_faces = []
                for part in parts[1:]:
                    if '//' in part:
                        v_idx = part.split('//')[0]
                        new_faces.append(v_idx)
                    else:
                        new_faces.append(part)
                outfile.write("f " + " ".join(new_faces) + "\n")
            elif line.startswith("v "):
                line_lst=line.split()
                line_lst=line_lst[:4]
                outfile.write(" ".join(line_lst) + "\n")
            else:
                outfile.write(line)
    print(f"[✓] Saved stripped .obj to: {output_path}")