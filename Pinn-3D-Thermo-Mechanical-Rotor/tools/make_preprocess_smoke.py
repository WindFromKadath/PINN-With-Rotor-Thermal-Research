from pathlib import Path
from thermo_rotor.training.run import sha256,write_json


def main():
    root=Path(__file__).resolve().parents[1]
    source=root/'data/raw/HeatPINNCheck.csv'
    target=root/'data/raw/heat_wide_smoke.csv'
    if target.exists():
        raise FileExistsError('Smoke subset already exists')
    rows=[]
    count=0
    with source.open(encoding='utf-8-sig') as f:
        for line in f:
            if line.startswith('% Nodes,'):
                rows.append('% Nodes,32\n')
            elif line.startswith('%'):
                rows.append(line)
            else:
                rows.append(line)
                count+=1
                if count==32:
                    break
    target.write_text(''.join(rows),encoding='utf-8')
    write_json(target.with_suffix('.json'),dict(source_file=source.name,source_sha256=sha256(source),
                    protocol='first 32 spatial rows, all original time columns; Nodes metadata adjusted',
                    rows=count,scope='COMSOL importer workflow diagnostic, no accuracy sampling claim',sha256=sha256(target)))


if __name__=='__main__':
    main()
