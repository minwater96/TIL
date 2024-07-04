 # git command

  > git 기본 명령어를 정리해봅시다.

## init
- 현재 위치에 `.git` 폴더를 생성

```bash
git init
```
## add
- working directory 에서 staging area로 올리는 실행문

```bash
git add .
```

## status
- 현재 git 상태 확인

```bash
git status
```

## commit
- staging area에 올라간 내용을 스냅샷 찍기
    - `-m` 옵션을 통해 커밋메세지를 바로 입력가능

```bash
git commit -m "first commit"
```

## remoto add
- 원격저장소의 주소를 저장하는 명령어

```bash
git remote add {remote_name} {remote_url}
```

## push
- 원격저장소로 브랜치를 업로드 하는 명령어

```bash
git push origin main
git push {remote_name} {branch_name}
```

## pull
- 원격저장소에서 브랜치를 다운로드 하는 명령어

```bash
git pull origin main
git pull {remote_name} {branch_name}
```
## branch

- 공용 원격저장소에서 기능단위 개별 작업을 위해 사용

1. 생성
```bash
git branch -c {branch_name}
```

2. 진입
```bash
git switch {branch_name}
```

3. 삭제
```bash
git branch -d {branch_name}
```