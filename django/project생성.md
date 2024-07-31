## basic setting

```markdown
# Django

1. 프로젝트생성
```bash
django-admin startproject {pjt_name}.
```

2. 가상환경 생성

```bash
python -m venv venv
```

3. 가상환경 활성화

```bash
source venv/bin/activate
```

4. 서버on (off: 'ctrl + c')
```bash
python manage.py runserver
```

5. 앱생성
```bash
django-admin startapp {app_name}
```

6. 앱등록 ('settings.py')
```python
INSTALLED_APPS = [
    ...
    '{app_name}'
]
```
```

## TIL

오늘은 django를 활용한 웹 만들기 기초를 배웠다.

기존 jupyter 설정과 비슷한 듯 다르게 여러가지 파일을 생성하고 활용하는 구조로 되어있었다.

먼저 `pip install django` 를 통해 project파일 터미널에 구조를 갖추고 시작하여,

위 코드를 통해 기본 세팅을 완료해야한다.

❗주의할점 : `startproject`를 할때는 파일 내에서 생성할 예정이기 때문에 뒤에 `.` 을 붙혀 

`manage.py` 의 위치를 잘 확인해야한다.

두번째는 `startapp`을 만들어 DB에 있는 자료를 불러올 환경을 구축해야한다.

위 코드를 통해 app을 만들고 거기에 맞춰 아까 만들어둔 pj 폴더에 있는 `settings.py` 를 통해

설정을 해서 두 폴더간의 다리를 만들어준다.

마지막으로 새롭게 만들어진 app폴더 안에 `templates` 파일을 새롭게 만들어 DB를 새롭게 구축한다.

❗주의할점 : 폴더명은 꼭 지켜야함!

만들어진 폴더 안에 `.html` 파일을 만들어 DB를 구축하면 끝!

#내맘대로TIL챌린지 #동아일보 #미디어 프론티어 #글로벌소프트웨어캠퍼스 #GSC신촌
글로벌소프트웨어캠퍼스와 동아일보가 함께 진행하는 챌린지입니다. 