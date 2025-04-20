FROM python:3.10

# hadolint ignore=DL3008
RUN apt-get update && \
	apt-get install --no-install-recommends -y \
	curl \
	fonts-ipafont-gothic \
	gcc \
	g++ \
	git \
	locales \
	make \
	neovim \
	pandoc \
	python3-dev \
	sudo \
	tzdata \
	vim \
	zsh && \
	apt-get clean && \
	rm -rf /var/lib/apt/lists/*

# poetryのインストール
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV PATH=/root/.local/bin:$PATH
RUN curl -sSL https://install.python-poetry.org | python - && \
	poetry config virtualenvs.create false

WORKDIR /app

# ライブラリのインストール
# poetry.lockが存在しないことを許容するため、./poetry.lock*としている
COPY ./pyproject.toml ./poetry.lock* /app/

RUN poetry install --no-root --no-interaction --no-ansi

COPY ./app.py /app/app.py
COPY ./src /app/src
COPY ./output /app/output

CMD ["python", "app.py"]
