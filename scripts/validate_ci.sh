#!/usr/bin/env bash
set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting CI/CD validation script${NC}"
echo "This script will validate your CI/CD configurations and dependencies"
echo

# Check for required tools
echo -e "${YELLOW}Checking for required tools...${NC}"

TOOLS=("python" "pip" "docker" "git")
MISSING_TOOLS=0

for tool in "${TOOLS[@]}"; do
    if ! command -v "$tool" &> /dev/null; then
        echo -e "${RED}❌ $tool is not installed${NC}"
        MISSING_TOOLS=$((MISSING_TOOLS+1))
    else
        echo -e "${GREEN}✓ $tool is installed${NC}"
    fi
done

if [ $MISSING_TOOLS -gt 0 ]; then
    echo -e "${RED}Please install the missing tools before continuing${NC}"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python --version | cut -d' ' -f2)
echo -e "${GREEN}Python version: $PYTHON_VERSION${NC}"

# Check if running in project directory
if [ ! -d ".git" ] || [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}This script must be run from the project root directory${NC}"
    exit 1
fi

# Install dev dependencies if not already installed
echo -e "${YELLOW}Installing development dependencies...${NC}"
if ! pip show pytest &> /dev/null; then
    pip install -r requirements-dev.txt
else
    echo -e "${GREEN}Development dependencies already installed${NC}"
fi

# Check for pre-commit
if ! command -v pre-commit &> /dev/null; then
    echo -e "${YELLOW}Installing pre-commit...${NC}"
    pip install pre-commit
else
    echo -e "${GREEN}✓ pre-commit is installed${NC}"
fi

# Install pre-commit hooks
echo -e "${YELLOW}Installing pre-commit hooks...${NC}"
pre-commit install

# Validate workflow files
echo -e "${YELLOW}Validating GitHub Actions workflow files...${NC}"
for workflow in .github/workflows/*.yml; do
    echo -n "Checking $workflow: "
    if python -c "import yaml; yaml.safe_load(open('$workflow'))"; then
        echo -e "${GREEN}✓ Valid YAML${NC}"
    else
        echo -e "${RED}❌ Invalid YAML${NC}"
        exit 1
    fi
done

# Run basic code quality checks
echo -e "${YELLOW}Running code quality checks...${NC}"

echo -n "Running black: "
if python -m black --check --quiet src tests; then
    echo -e "${GREEN}✓ Passed${NC}"
else
    echo -e "${RED}❌ Failed${NC}"
    echo "Run 'black src tests' to fix formatting"
fi

echo -n "Running flake8: "
if python -m flake8 src tests; then
    echo -e "${GREEN}✓ Passed${NC}"
else
    echo -e "${RED}❌ Failed${NC}"
fi

echo -n "Running isort: "
if python -m isort --check-only --profile black src tests; then
    echo -e "${GREEN}✓ Passed${NC}"
else
    echo -e "${RED}❌ Failed${NC}"
    echo "Run 'isort src tests' to fix import order"
fi

echo -n "Running mypy: "
if python -m mypy src; then
    echo -e "${GREEN}✓ Passed${NC}"
else
    echo -e "${RED}❌ Failed${NC}"
fi

# Run security checks
echo -e "${YELLOW}Running security checks...${NC}"

echo -n "Running bandit: "
if python -m bandit -r src -c pyproject.toml -q; then
    echo -e "${GREEN}✓ Passed${NC}"
else
    echo -e "${RED}❌ Failed${NC}"
fi

# Run unit tests
echo -e "${YELLOW}Running unit tests...${NC}"
python -m pytest tests/unit -v

# Check Docker
echo -e "${YELLOW}Validating Dockerfile...${NC}"
if docker build -t hipo:test -f Dockerfile . --target build; then
    echo -e "${GREEN}✓ Dockerfile is valid${NC}"
    docker rmi hipo:test &> /dev/null || true
else
    echo -e "${RED}❌ Dockerfile build failed${NC}"
fi

# Validate Kubernetes configurations
echo -e "${YELLOW}Validating Kubernetes configurations...${NC}"
if command -v kubectl &> /dev/null; then
    for config in k8s/*/*.yaml; do
        echo -n "Checking $config: "
        if kubectl apply --dry-run=client -f "$config" &> /dev/null; then
            echo -e "${GREEN}✓ Valid${NC}"
        else
            echo -e "${RED}❌ Invalid${NC}"
        fi
    done
else
    echo -e "${YELLOW}kubectl not found, skipping Kubernetes validation${NC}"
fi

echo -e "${GREEN}CI/CD validation completed${NC}"
echo "Your environment is ready for development!"
echo "Remember to run 'pre-commit run --all-files' before committing changes"