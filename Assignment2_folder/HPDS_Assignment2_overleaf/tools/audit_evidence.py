"""Audit archived measurements offline without extracting or executing archive contents."""
import csv, hashlib, io, json, math, statistics, tarfile
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / 'data'
OUT.mkdir(exist_ok=True)
reports = []
all_data = {}
for archive in sorted(BASE.glob('*evidence*.tar.gz')):
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert digest == Path(str(archive)+'.sha256').read_text().split()[0]
    with tarfile.open(archive) as t:
        files = {m.name: t.extractfile(m).read() for m in t.getmembers() if m.isfile()}
    root = next(iter(files)).split('/')[0]+'/'
    files = {k[len(root):]: v for k,v in files.items()}
    def csvrows(name): return list(csv.DictReader(io.StringIO(files[name].decode())))
    manifest = csvrows('manifest.csv')
    for entry in manifest:
        assert hashlib.sha256(files[entry['bundle_path']]).hexdigest() == entry['sha256']
    if 'mpi-cuda' in archive.name:
        import re
        checked = defaultdict(int)
        for name, value in files.items():
            if 'mpi-verify.' not in name or not name.endswith('.txt') or '-serial' in name: continue
            parent, filename = name.rsplit('/',1)
            dataset = filename.split('-')[0]
            assert value == files[parent+'/'+dataset+'-serial.txt']
            log = files[name[:-4]+'.log'].decode()
            ranks = [dict(re.findall(r'(host|pci|queries|start)=([^ ]+)', line)) for line in log.splitlines() if line.startswith('RANK ')]
            assert len({(r['host'],r['pci']) for r in ranks}) == len(ranks)
            end=0
            for r in sorted(ranks,key=lambda r:int(r['start'])):
                assert int(r['start'])==end; end+=int(r['queries'])
            assert end==len(value.splitlines())
            checked[re.search(r'job(\d+)',name).group(1)]+=1
        assert dict(checked)=={'3382992':36,'3382997':72,'3382990':36}
        (OUT/'mpi_verification_audit.json').write_text(json.dumps(dict(checked),indent=2)+'\n')
        print('MPI predictions, unique host/device mappings and query coverage verified',dict(checked))
    record = dict(archive=archive.name, sha256=digest, manifest_files=len(manifest), benchmark_rows=0, groups=0)
    for rawname in [n for n in files if n.endswith('/raw.csv')]:
        parent = rawname.rsplit('/',1)[0]
        raw = csvrows(rawname)
        cpu = 'program' in raw[0]
        kind = 'program' if cpu else 'strategy'; param = 'workers' if cpu else 'block_size'
        groups = defaultdict(list)
        def related(stem, suffix, category):
            possibilities = [parent+'/'+stem+suffix, parent+'/'+category+'/'+stem+suffix]
            return next(files[n] for n in possibilities if n in files)
        for r in raw:
            key = (r['dataset'],r[kind],r[param]); groups[key].append(r)
            stem = f"{r['dataset']}-{r[kind]}-{'w' if cpu else 'b'}{r[param]}-r{r['rep']}"
            ref = f"{r['dataset']}-serial-{'w1' if cpu else 'b0'}-r1"
            predictions = related(stem,'.predictions.txt','predictions')
            assert predictions == related(ref,'.predictions.txt','predictions')
            logs = related(stem,'.log','logs').decode()
            records = [l for l in logs.splitlines() if l.startswith('RESULT ')]
            assert len(records)==1
            result = dict(x.split('=',1) for x in records[0].split()[1:])
            assert result['strategy']==r[kind]
            for field in ['end_to_end_ms','accuracy']+(['kernel_ms'] if not cpu else []):
                assert result.get(field,'')==r.get(field,'')
            assert r['k']=='3' and r['predictions_match']=='yes'
            text = files['source/datasets/'+r['dataset']+'-test.arff'].decode()
            lines=text.split('@data')[-1].strip().splitlines()
            labels=[int(float(l.split(',')[-1])) for l in lines if l.strip() and not l.startswith('%')]
            pred=list(map(int,predictions.split()));assert len(pred)==len(labels)
            accuracy=100*sum(a==b for a,b in zip(pred,labels))/len(labels)
            assert abs(accuracy-float(r['accuracy']))<1e-5
        for group in groups.values(): assert sorted(r['rep'] for r in group)==['1','2','3']
        summary=csvrows(parent+'/summary.csv');assert len(summary)==len(groups)
        for r in summary:
            key=(r['dataset'],r[kind],r[param]); ts=[float(v['end_to_end_ms']) for v in groups[key]]
            mean=statistics.mean(ts)
            serial=statistics.mean(float(v['end_to_end_ms']) for v in groups[(r['dataset'],'serial','1' if cpu else '0')])
            expected={'mean_ms':mean,'min_ms':min(ts),'max_ms':max(ts), 'speedup_vs_serial' if cpu else 'serial_speedup':serial/mean}
            if cpu and r[kind]!='serial':
                one=statistics.mean(float(v['end_to_end_ms']) for v in groups[(r['dataset'],r[kind],'1')])
                expected.update(speedup_vs_1_worker=one/mean,efficiency_vs_1_worker=one/mean/int(r[param]))
            for field,value in expected.items(): assert math.isclose(float(r[field]),value,rel_tol=1e-9,abs_tol=1e-9)
        if parent+'/kernel-summary.csv' in files:
            for r in csvrows(parent+'/kernel-summary.csv'):
                g=groups[(r['dataset'],r.get('strategy','global'),r['block_size'])]
                assert abs(float(r['mean_kernel_ms'])-statistics.mean(float(v['kernel_ms']) for v in g))<1e-6
        record['benchmark_rows']+=len(raw);record['groups']+=len(groups)
        tag=archive.name.replace('-evidence','').replace('.tar.gz','')
        all_data[tag]={'raw':raw,'summary':summary,'archive':archive.name,'raw_path':rawname}
        for kindfile in ['raw.csv','summary.csv']:
            (OUT/(tag+'-'+kindfile)).write_bytes(files[parent+'/'+kindfile])
    # Match repository report-data copies to actual archived bytes, checked below globally.
    reports.append(record)
    print(record)
    all_data.setdefault('_archived_csv_hashes',[]).extend(hashlib.sha256(b).hexdigest() for n,b in files.items() if n.endswith('.csv'))
for p in (BASE/'report_data').glob('*.csv'):
    assert hashlib.sha256(p.read_bytes()).hexdigest() in all_data['_archived_csv_hashes'],p
all_data.pop('_archived_csv_hashes')
(OUT/'audited_data.json').write_text(json.dumps(all_data,indent=2)+'\n')
(OUT/'audit.json').write_text(json.dumps(reports,indent=2)+'\n')
print('All report CSVs match archive contents. Total benchmark rows',sum(r['benchmark_rows'] for r in reports))
