# 반영 방법

이 패키지는 `SEJUNHONG/MY_threestudio`의 `c894bfc`를 기준으로 수정했습니다. 원격 저장소에는 아직 push하지 않았습니다.

1. 기존 저장소를 clone하고 `integration-repair` 같은 새 브랜치를 생성합니다.
2. 변경 파일 패키지를 저장소 최상위에 덮어씁니다. 새로운 파일과 폴더도 함께 복사합니다.
3. `git diff`에서 기존 작업과 충돌이 없는지 확인합니다. 특히 원격 코드가 기준 커밋 이후 변경되었다면 무조건 덮어쓰지 마세요.
4. CPU 테스트 후, 사용 가능한 GPU에서 preflight와 2-step smoke 실행을 확인합니다.
5. 변경 파일만 커밋하고 PR로 반영합니다. 모델 가중치·출력·개인 환경은 커밋하지 않습니다.

```bash
git switch -c integration-repair
# 변경 파일 복사
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
git status --short
git diff --stat
# 파일별 검토 후 git add / commit / push
```

`README.md`와 `README.ko.md`는 소개 문서이며, 실험적으로 검증된 기능과 미검증 범위를 구분합니다. 원본 라이선스·개별 component notice를 유지하세요. 추가한 파인튜닝 스크립트는 과거 수행한 학습의 재현 증거로 표현하지 않습니다.
