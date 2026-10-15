"""빌드 후 README와 라이선스를 포함한 Windows 배포 ZIP을 만든다."""
import hashlib
import importlib.metadata
import shutil
import sys
import zipfile
from pathlib import Path

root=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from engineering_math import VERSION
folder=root/'dist/EngineeringMathStudio'
if not (folder/'EngineeringMathStudio.exe').is_file():
    raise FileNotFoundError('build_windows.ps1을 먼저 실행해.')
smoke=folder/'smoke-test.json'
if smoke.exists():
    smoke.replace(root/'build/packaged-smoke-test.json')
shutil.copy2(root/'README.md',folder/'README.md')
shutil.copy2(root/'verify_windows.ps1',folder/'verify_windows.ps1')
shutil.copy2(root/'VALIDATION.md',folder/'VALIDATION.md')
GUIDE = """공업수학 학습 스튜디오 v{version}

EngineeringMathStudio.exe를 더블클릭하십시오.
Python을 별도로 설치할 필요가 없습니다. _internal을 포함한 전체 폴더를 함께 보관하십시오.

내용: Kreyszig 『공업수학 10판 하』 11장 Fourier 해석, 12장 편미분방정식 (교재 본문 예제 프리셋 53개)

사용법
- 왼쪽 목차: 장 → 절 → 교재 예제. 예제를 누르면 바로 계산합니다. 위쪽 검색창으로 절·예제 번호를 찾습니다.
- 각 절의 "사용자 지정": 함수와 매개변수를 직접 입력합니다. 길이 변수는 l입니다 (예: sin(pi*x/l)).
- 값을 바꾸면 자동으로 다시 계산하고, 이전 결과를 흐린 파선으로 겹쳐 보여 줍니다.
- 그래프 아래 재생 막대로 시간·차수에 따른 변화를 재생합니다. 변화 과정 탭에서는 현·막대·막의 움직임을 봅니다.
- 그래프 위 Ctrl+휠 확대, Shift+휠 좌우 이동, 더블클릭 원래대로. 보기 메뉴에서 글자 크기 조절.
- 파일 메뉴: 설정(JSON) 저장·불러오기, 결과 CSV 묶음 저장. 수치 표는 Ctrl+C로 전체 자릿수 복사.

실행 검사: verify_windows.ps1을 PowerShell에서 실행하면 프로그램 자체 검사를 수행합니다.
현재 Windows x64 PC에서 실행 검사 완료. 다른 PC의 동작은 별도 확인이 필요합니다.
"""
(folder/'실행 안내.txt').write_text(GUIDE.format(version=VERSION), encoding='utf-8-sig')
license_folder=folder/'licenses'
license_folder.mkdir(exist_ok=True)
packages=['PySide6','PySide6_Essentials','PySide6_Addons','shiboken6','scipy','numpy','matplotlib','sympy','mpmath','pillow','packaging','kiwisolver','contourpy',
          'pyparsing','python-dateutil','six','cycler','fonttools','psutil','charset-normalizer','jinja2','markupsafe']
versions=[]
for package in packages:
    distribution=importlib.metadata.distribution(package)
    versions.append(f'{package}=={distribution.version}')
    for entry in distribution.files or []:
        if any(part.lower().startswith(('license','copying','notice')) for part in entry.parts):
            source=Path(distribution.locate_file(entry))
            if source.is_file():
                target=license_folder/package/Path(str(entry))
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(source,target)
for name in ['LICENSE.txt','LICENSE']:
    source=Path(sys.base_prefix)/name
    if source.is_file(): shutil.copy2(source,license_folder/f'Python-{name}')
(folder/'versions.txt').write_text('\n'.join(versions),encoding='utf-8')
archive_path=root/f'dist/EngineeringMathStudio-v{VERSION}-windows-x64.zip'
with zipfile.ZipFile(archive_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
    for file in folder.rglob('*'):
        # 자체 검사가 남긴 보고서·화면 캡처는 배포 ZIP에 넣지 않습니다.
        if file.is_file() and not (file.suffix == '.png' or file.name == 'validation-report.json'):
            archive.write(file,file.relative_to(folder.parent))
digest=hashlib.sha256(archive_path.read_bytes()).hexdigest()
archive_path.with_suffix('.sha256').write_text(f'{digest}  {archive_path.name}\n',encoding='ascii')
print(f'Release ZIP: {archive_path.name}, {archive_path.stat().st_size/1024**2:.1f} MiB')
