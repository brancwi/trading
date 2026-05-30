#!/usr/bin/env bash
# ============================================================
# Dev Manager — Gestion des services en mode développement
# ============================================================
# Usage:
#   ./scripts/dev_manager.sh start    # Démarre tous les services
#   ./scripts/dev_manager.sh stop     # Arrête tous les services
#   ./scripts/dev_manager.sh status   # État des services
#   ./scripts/dev_manager.sh logs     # Logs temps réel (tous)
#   ./scripts/dev_manager.sh logs mcp # Logs temps réel (MCP uniquement)
#   ./scripts/dev_manager.sh restart mcp  # Redémarre MCP
# ============================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$PROJECT_DIR/logs"
PID_DIR="/tmp/trading-dev"

mkdir -p "$LOG_DIR" "$PID_DIR"

# Couleurs
GRN='\033[0;32m'
RED='\033[0;31m'
YEL='\033[1;33m'
NC='\033[0m'

# --- Helpers ------------------------------------------------

pid_file() { echo "$PID_DIR/${1}.pid"; }
log_file() { echo "$LOG_DIR/${1}.log"; }

get_pid() {
    local f="$(pid_file "$1")"
    [[ -f "$f" ]] && cat "$f" 2>/dev/null || echo ""
}

is_running() {
    local pid="$(get_pid "$1")"
    [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

kill_service() {
    local name="$1"
    local pid="$(get_pid "$name")"
    if [[ -n "$pid" ]]; then
        kill "$pid" 2>/dev/null || true
        # Wait for process to die
        for i in {1..10}; do
            if ! kill -0 "$pid" 2>/dev/null; then
                break
            fi
            sleep 0.5
        done
        # Force kill if still running
        kill -9 "$pid" 2>/dev/null || true
        rm -f "$(pid_file "$name")"
    fi
}

# --- Commands -----------------------------------------------

cmd_start() {
    echo -e "${GRN}▶ Démarrage des services de développement...${NC}"

    # 1. PostgreSQL (Docker)
    if docker ps --filter name=trading-postgres --format '{{.Names}}' | grep -q .; then
        echo -e "  ${GRN}✓${NC} PostgreSQL déjà lancé"
    else
        echo -e "  ${YEL}→${NC} Démarrage PostgreSQL..."
        cd "$PROJECT_DIR" && docker compose up -d postgres
        sleep 2
        echo -e "  ${GRN}✓${NC} PostgreSQL lancé"
    fi

    # 2. MCP Server
    if is_running mcp-server; then
        echo -e "  ${GRN}✓${NC} MCP Server déjà lancé (PID: $(get_pid mcp-server))"
    else
        echo -e "  ${YEL}→${NC} Démarrage MCP Server..."
        cd "$PROJECT_DIR"
        nohup uv run python scripts/run_mcp_server.py > "$(log_file mcp-server)" 2>&1 &
        echo $! > "$(pid_file mcp-server)"
        sleep 2
        echo -e "  ${GRN}✓${NC} MCP Server lancé (PID: $(get_pid mcp-server))"
    fi

    # 3. API FastAPI
    if is_running api-server; then
        echo -e "  ${GRN}✓${NC} API Server déjà lancé (PID: $(get_pid api-server))"
    else
        echo -e "  ${YEL}→${NC} Démarrage API Server..."
        cd "$PROJECT_DIR"
        nohup uv run python scripts/run_api.py > "$(log_file api-server)" 2>&1 &
        echo $! > "$(pid_file api-server)"
        sleep 2
        echo -e "  ${GRN}✓${NC} API Server lancé (PID: $(get_pid api-server))"
    fi

    # 4. Event Listener
    if is_running listener; then
        echo -e "  ${GRN}✓${NC} Listener déjà lancé (PID: $(get_pid listener))"
    else
        echo -e "  ${YEL}→${NC} Démarrage Listener..."
        cd "$PROJECT_DIR"
        nohup uv run python scripts/run_listener.py > "$(log_file listener)" 2>&1 &
        echo $! > "$(pid_file listener)"
        sleep 2
        echo -e "  ${GRN}✓${NC} Listener lancé (PID: $(get_pid listener))"
    fi

    echo ""
    echo -e "${GRN}Services accessibles :${NC}"
    echo "  MCP Server   → http://localhost:8001/sse"
    echo "  API Docs     → http://localhost:8000/docs"
    echo "  API Health   → http://localhost:8000/health"
    echo ""
    echo -e "${YEL}Pour voir les logs :${NC} ./scripts/dev_manager.sh logs"
}

cmd_stop() {
    echo -e "${RED}▶ Arrêt des services...${NC}"
    for svc in listener api-server mcp-server; do
        if is_running "$svc"; then
            echo -e "  ${YEL}→${NC} Arrêt $svc (PID: $(get_pid $svc))..."
            kill_service "$svc"
            echo -e "  ${GRN}✓${NC} $svc arrêté"
        else
            echo -e "  ${GRN}✓${NC} $svc déjà arrêté"
        fi
    done
    echo -e "${GRN}Tous les services arrêtés.${NC}"
}

cmd_status() {
    echo -e "${YEL}▶ État des services${NC}"
    echo ""
    # PostgreSQL
    if docker ps --filter name=trading-postgres --format '{{.Names}}' | grep -q .; then
        echo -e "  ${GRN}●${NC} PostgreSQL    $(docker ps --filter name=trading-postgres --format '{{.Status}}' | head -1)"
    else
        echo -e "  ${RED}●${NC} PostgreSQL    arrêté"
    fi
    # Services
    for svc in mcp-server api-server listener; do
        if is_running "$svc"; then
            echo -e "  ${GRN}●${NC} $svc    running (PID: $(get_pid $svc))"
        else
            echo -e "  ${RED}●${NC} $svc    arrêté"
        fi
    done
    echo ""
    echo -e "${YEL}Ports utilisés :${NC}"
    ss -tlnp 2>/dev/null | grep -E ':8000|:8001|:5432' || netstat -tlnp 2>/dev/null | grep -E ':8000|:8001|:5432' || echo "  (commande ss/netstat non disponible)"
}

cmd_logs() {
    local svc="${1:-}"
    if [[ -z "$svc" ]]; then
        echo -e "${YEL}▶ Logs temps réel (tous les services — Ctrl+C pour quitter)${NC}"
        tail -f "$(log_file mcp-server)" "$(log_file api-server)" "$(log_file listener)" 2>/dev/null
    else
        local logf="$(log_file "$svc")"
        if [[ -f "$logf" ]]; then
            echo -e "${YEL}▶ Logs $svc (Ctrl+C pour quitter)${NC}"
            tail -f "$logf"
        else
            echo -e "${RED}✗${NC} Pas de logs pour '$svc'"
        fi
    fi
}

cmd_restart() {
    local svc="${1:-}"
    if [[ -z "$svc" ]]; then
        echo -e "${RED}Usage: restart <service>${NC}"
        echo "  Services: mcp-server, api-server, listener"
        exit 1
    fi
    echo -e "${YEL}▶ Redémarrage $svc...${NC}"
    kill_service "$svc"
    sleep 1
    case "$svc" in
        mcp-server)
            cd "$PROJECT_DIR"
            nohup uv run python scripts/run_mcp_server.py > "$(log_file mcp-server)" 2>&1 &
            ;;
        api-server)
            cd "$PROJECT_DIR"
            nohup uv run python scripts/run_api.py > "$(log_file api-server)" 2>&1 &
            ;;
        listener)
            cd "$PROJECT_DIR"
            nohup uv run python scripts/run_listener.py > "$(log_file listener)" 2>&1 &
            ;;
        *)
            echo -e "${RED}Service inconnu: $svc${NC}"
            exit 1
            ;;
    esac
    echo $! > "$(pid_file "$svc")"
    sleep 2
    if is_running "$svc"; then
        echo -e "  ${GRN}✓${NC} $svc redémarré (PID: $(get_pid $svc))"
    else
        echo -e "  ${RED}✗${NC} Échec du redémarrage de $svc"
    fi
}

# --- Main ---------------------------------------------------

case "${1:-}" in
    start)
        cmd_start
        ;;
    stop)
        cmd_stop
        ;;
    restart)
        cmd_restart "${2:-}"
        ;;
    status|stat)
        cmd_status
        ;;
    logs|log)
        cmd_logs "${2:-}"
        ;;
    *)
        echo "Dev Manager — Trading Engine V4.2"
        echo ""
        echo "Usage:"
        echo "  $0 start              Démarre tous les services"
        echo "  $0 stop               Arrête tous les services"
        echo "  $0 restart <service>  Redémarre un service (mcp-server|api-server|listener)"
        echo "  $0 status             État des services"
        echo "  $0 logs [service]     Logs temps réel (service optionnel)"
        echo ""
        ;;
esac
