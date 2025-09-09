import os
import streamlit as st
import httpx
import json
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="HFT Neurosymbolic Dashboard", layout="wide")
st.title("🧠 HFT Neurosymbolic AI - Dashboard")

# Inject lightweight theme and utility CSS
st.markdown(
    """
    <style>
      :root {
        --bg: #0b1220; 
        --card: #121b2e; 
        --text: #e6eefc; 
        --muted: #9fb3d1; 
        --primary: #3b82f6; 
        --success: #22c55e; 
        --warn: #f59e0b; 
        --danger: #ef4444;
        --accent: #8b5cf6;
      }
      .app-bg { background: var(--bg); }
      .kpi-card { 
        background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
        border: 1px solid rgba(255,255,255,0.06); 
        padding: 16px; border-radius: 12px; 
        box-shadow: 0 6px 18px rgba(0,0,0,0.25);
      }
      .kpi-title { color: var(--muted); font-size: 12px; letter-spacing: .06em; text-transform: uppercase; }
      .kpi-value { color: var(--text); font-size: 26px; font-weight: 700; }
      .kpi-sub { color: var(--muted); font-size: 12px; }
      .badge { 
        display: inline-block; padding: 4px 10px; border-radius: 999px; font-size: 12px; 
        border: 1px solid rgba(255,255,255,0.08); background: rgba(255,255,255,0.04); color: var(--text);
      }
      .badge.success { border-color: rgba(34,197,94,.35); background: rgba(34,197,94,.08); color: #b7f7c9; }
      .badge.warn { border-color: rgba(245,158,11,.35); background: rgba(245,158,11,.08); color: #ffe2b3; }
      .badge.danger { border-color: rgba(239,68,68,.35); background: rgba(239,68,68,.08); color: #ffc1c1; }
      .soft-button a { 
        text-decoration: none; color: var(--text); background: linear-gradient(180deg, rgba(59,130,246,.25), rgba(59,130,246,.15));
        border: 1px solid rgba(59,130,246,.35); padding: 10px 14px; border-radius: 10px; display: inline-flex; gap: 8px; align-items: center;
      }
      .soft-button a:hover { filter: brightness(1.1); }
      .section-title { font-size: 18px; font-weight: 700; color: var(--text); margin-bottom: 6px; }
      .muted { color: var(--muted); }
      .result-card { background: var(--card); border-radius: 12px; border: 1px solid rgba(255,255,255,0.06); padding: 16px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# External base used for browser links (host ports)
api_external = os.getenv("API_EXTERNAL", "http://localhost:8001")
# Internal base used for server-side health check (same container)
api_internal = os.getenv("API_INTERNAL", "http://127.0.0.1:8000")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Overview", "Trading Signals", "Signal Runner", "Reasoning Traces", "System Health"])

if page == "Overview":
    # Header actions
    c1, c2, c3, c4 = st.columns([1,1,1,1])
    with c1:
        st.markdown(f"<span class='soft-button'><a href='{api_external}/docs' target='_blank'>📜 OpenAPI Docs</a></span>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<span class='soft-button'><a href='{api_external}/metrics' target='_blank'>📈 Prometheus</a></span>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<span class='soft-button'><a href='http://localhost:8501' target='_blank'>🖥️ Dashboard</a></span>", unsafe_allow_html=True)
    with c4:
        st.markdown(f"<span class='soft-button'><a href='{api_external}/redoc' target='_blank'>📘 ReDoc</a></span>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>System Snapshot</div>", unsafe_allow_html=True)

    # Fetch health + status
    components_status = {}
    overall_status = "unknown"
    metrics_summary = {
        "signals": "-",
        "ai_predictions": "-",
        "avg_pred_ms": "-",
        "avg_reason_ms": "-",
    }
    try:
        hr = httpx.get(f"{api_internal}/health", timeout=3.0)
        if hr.status_code == 200:
            h = hr.json()
            components_status = h.get("components", {})
            overall_status = h.get("status", "unknown")
    except Exception:
        pass

    try:
        sr = httpx.get(f"{api_internal}/api/v1/system/status", timeout=4.0)
        if sr.status_code == 200:
            s = sr.json().get("metrics", {})
            metrics_summary["signals"] = s.get("trading", {}).get("signals_generated", 0)
            metrics_summary["ai_predictions"] = s.get("ai", {}).get("predictions_made", 0)
            metrics_summary["avg_pred_ms"] = s.get("ai", {}).get("avg_prediction_time_ms", 0.0)
            metrics_summary["avg_reason_ms"] = s.get("symbolic", {}).get("avg_reasoning_time_ms", 0.0)
    except Exception:
        pass

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown("<div class='kpi-card'><div class='kpi-title'>Overall Status</div>", unsafe_allow_html=True)
        badge_class = "success" if overall_status == "healthy" else ("warn" if overall_status == "degraded" else "danger")
        st.markdown(f"<div class='kpi-value'><span class='badge {badge_class}'>{overall_status.title()}</span></div></div>", unsafe_allow_html=True)
    with k2:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Signals Generated</div><div class='kpi-value'>{metrics_summary['signals']}</div><div class='kpi-sub'>Total to date</div></div>", unsafe_allow_html=True)
    with k3:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>AI Predictions</div><div class='kpi-value'>{metrics_summary['ai_predictions']}</div><div class='kpi-sub'>Since start</div></div>", unsafe_allow_html=True)
    with k4:
        st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Latency (ms)</div><div class='kpi-value'>{metrics_summary['avg_pred_ms']}/{metrics_summary['avg_reason_ms']}</div><div class='kpi-sub'>Predict / Reason</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Component Health</div>", unsafe_allow_html=True)
    hc1, hc2, hc3 = st.columns(3)
    comp_items = list(components_status.items())
    for i, (name, ok) in enumerate(comp_items):
        col = [hc1, hc2, hc3][i % 3]
        status_class = "success" if ok else "danger"
        col.markdown(f"<div class='kpi-card'><div class='kpi-title'>{name.replace('_',' ').title()}</div><div class='kpi-value'><span class='badge {status_class}'>{'Healthy' if ok else 'Unhealthy'}</span></div></div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Tips</div><span class='muted'>Use the sidebar to navigate. Generate signals to populate reasoning traces and metrics.</span>", unsafe_allow_html=True)

elif page == "Trading Signals":
    st.subheader("📈 Trading Signal Generator")
    col1, col2 = st.columns(2)
    with col1:
        symbol = st.selectbox("Symbol", ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "NVDA"]) 
        timeframe = st.selectbox("Timeframe", ["daily", "hourly", "minute"]) 
        strategy = st.selectbox("Strategy", ["default", "neurosymbolic", "momentum", "mean_reversion", "rule_only"]) 
    with col2:
        if st.button("🚀 Generate Signal", type="primary"):
            with st.spinner("Generating trading signal..."):
                try:
                    response = httpx.post(
                        f"{api_internal}/api/v1/trading/signal",
                        json={
                            "symbol": symbol,
                            "timeframe": timeframe,
                            "strategy": strategy
                        },
                        timeout=12.0
                    )
                    if response.status_code == 200:
                        signal_data = response.json()

                        with st.container():
                            st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                            top_c1, top_c2, top_c3 = st.columns([2,1,1])
                            with top_c1:
                                action = signal_data.get("signal", {}).get("action", "unknown").upper()
                                color = "success" if action == "BUY" else ("danger" if action == "SELL" else "warn")
                                st.markdown(f"<span class='badge {color}'>Signal</span> <span style='margin-left:8px;font-weight:700;font-size:20px'>{action}</span>", unsafe_allow_html=True)
                                st.caption(signal_data.get("signal", {}).get("reasoning", "N/A"))
                            with top_c2:
                                st.metric("Final Confidence", f"{signal_data.get('confidence', 0):.2f}")
                            with top_c3:
                                st.metric("Signal Time (ms)", f"{signal_data.get('signal', {}).get('signal_time_ms', 0):.1f}")

                            tabs = st.tabs(["Overview", "AI Prediction", "Symbolic Analysis", "Raw JSON"]) 
                            with tabs[0]:
                                st.write({k: signal_data[k] for k in ["symbol", "timestamp"] if k in signal_data})
                            with tabs[1]:
                                ai_pred = signal_data.get("ai_prediction", {})
                                if ai_pred:
                                    st.metric("AI Confidence", f"{ai_pred.get('ensemble', {}).get('confidence', 0):.2f}")
                                    st.write({k: ai_pred[k] for k in ["model_count", "prediction_time_ms"] if k in ai_pred})
                                else:
                                    st.info("No AI prediction details available")
                            with tabs[2]:
                                symbolic = signal_data.get("symbolic_analysis", {})
                                if symbolic:
                                    analysis = symbolic.get("analysis", {})
                                    st.write("Market Regime:", analysis.get("market_regime", {}))
                                    st.write("Technical Signals:", analysis.get("technical_signals", {}))
                                else:
                                    st.info("No symbolic analysis details available")
                            with tabs[3]:
                                st.code(json.dumps(signal_data, indent=2))

                            st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.error(f"❌ Error: {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"❌ Failed to generate signal: {e}")

elif page == "Signal Runner":
	st.subheader("🏃‍♂️ Trading Signal Runner")
	
	# Signal runner configuration
	col1, col2 = st.columns(2)
	
	with col1:
		st.subheader("📋 Configuration")
		
		# Symbol selection
		symbols_input = st.text_area(
			"Symbols (one per line)", 
			value="AAPL\nTSLA\nMSFT\nGOOGL\nAMZN",
			height=100
		)
		symbols = [s.strip() for s in symbols_input.split('\n') if s.strip()]
		
		# Strategy selection
		strategy = st.selectbox("Strategy", ["default", "neurosymbolic", "momentum", "mean_reversion", "rule_only"]) 
		
		# Timeframe selection
		timeframe = st.selectbox("Timeframe", ["daily", "hourly", "minute"]) 
		
		# Run options
		auto_refresh = st.checkbox("Auto-refresh (every 30 seconds)", value=False)
		max_signals = st.slider("Max signals to display", 1, 50, 10)
	
	with col2:
		st.subheader("🎮 Controls")
		
		col2_1, col2_2 = st.columns(2)
		
		with col2_1:
			if st.button("🚀 Run Signals", type="primary"):
				st.session_state.run_signals = True
		
		with col2_2:
			if st.button("⏹️ Stop Runner"):
				st.session_state.run_signals = False
		
		# Status
		if st.session_state.get('run_signals', False):
			st.success("🟢 Signal runner is active")
		else:
			st.info("🔴 Signal runner is stopped")
	
	# Signal results
	st.subheader("📊 Signal Results")
	
	if st.session_state.get('run_signals', False) or st.button("🔄 Refresh Results"):
		with st.spinner("Generating signals..."):
			signal_results = []
			
			for symbol in symbols[:max_signals]:
				try:
					response = httpx.post(
						f"{api_internal}/api/v1/trading/signal",
						json={
							"symbol": symbol,
							"timeframe": timeframe,
							"strategy": strategy
						},
						timeout=10.0
					)
					
					if response.status_code == 200:
						signal_data = response.json()
						signal = signal_data.get("signal", {})
						reasoning_text = signal.get("reasoning", "N/A")

						# Derive action more robustly to avoid UNKNOWN
						action_raw = signal.get("action")
						if not action_raw:
							# Try from symbolic recommendation
							action_raw = (
								signal_data.get("symbolic_analysis", {})
								.get("analysis", {})
								.get("trading_recommendation", {})
								.get("action")
							)
						if not action_raw:
							# Fallback from reasoning keywords
							lower_reason = (reasoning_text or "").lower()
							if "risk" in lower_reason or "compliance" in lower_reason:
								action_raw = "hold"
							else:
								action_raw = "wait"

						# Regime and technical for extra context
						analysis = signal_data.get("symbolic_analysis", {}).get("analysis", {})
						regime = analysis.get("market_regime", {}).get("regime", "unknown")
						technical = analysis.get("technical_signals", {}).get("signal", "unknown")

						signal_results.append({
							"Symbol": symbol,
							"Action": str(action_raw).upper(),
							"Confidence": f"{signal.get('confidence', 0):.2f}",
							"Reasoning": reasoning_text,
							"AI Confidence": f"{signal_data.get('ai_prediction', {}).get('ensemble', {}).get('confidence', 0):.2f}",
							"Symbolic Confidence": f"{analysis.get('trading_recommendation', {}).get('confidence', 0):.2f}",
							"Regime": regime,
							"Technical": str(technical).upper(),
							"Timestamp": signal_data.get("timestamp", "N/A")
						})
					else:
						signal_results.append({
							"Symbol": symbol,
							"Action": "ERROR",
							"Confidence": "0.00",
							"Reasoning": f"HTTP {response.status_code}",
							"AI Confidence": "0.00",
							"Symbolic Confidence": "0.00",
							"Regime": "unknown",
							"Technical": "unknown",
							"Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
						})
						
				except Exception as e:
					signal_results.append({
						"Symbol": symbol,
						"Action": "ERROR",
						"Confidence": "0.00",
						"Reasoning": str(e)[:50],
						"AI Confidence": "0.00",
						"Symbolic Confidence": "0.00",
						"Regime": "unknown",
						"Technical": "unknown",
						"Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
					})
			
			# Display results
			if signal_results:
				df = pd.DataFrame(signal_results)
				
				# Color code actions
				def color_action(val):
					if val == "BUY":
						return "background-color: #d4edda; color: #155724"
					elif val == "SELL":
						return "background-color: #f8d7da; color: #721c24"
					elif val in ("HOLD", "WAIT"):
						return "background-color: #fff3cd; color: #856404"
					elif val == "ERROR":
						return "background-color: #f5c6cb; color: #721c24"
					return ""
				
				styled_df = df.style.applymap(color_action, subset=['Action'])
				st.dataframe(styled_df, use_container_width=True)
				
				# Summary statistics
				st.subheader("📈 Summary Statistics")
				col1, col2, col3, col4 = st.columns(4)
				
				with col1:
					total_signals = len(signal_results)
					st.metric("Total Signals", total_signals)
				
				with col2:
					buy_signals = len([s for s in signal_results if s["Action"] == "BUY"])
					st.metric("Buy Signals", buy_signals)
				
				with col3:
					sell_signals = len([s for s in signal_results if s["Action"] == "SELL"])
					st.metric("Sell Signals", sell_signals)
				
				with col4:
					hold_signals = len([s for s in signal_results if s["Action"] in ("HOLD", "WAIT")])
					st.metric("Hold/Wait", hold_signals)
				
				# Average confidence
				try:
					nonzero = [float(s["Confidence"]) for s in signal_results if s["Confidence"] not in ("0.00", "0", 0)]
					avg_confidence = sum(nonzero) / len(nonzero) if nonzero else 0.0
					st.metric("Average Confidence", f"{avg_confidence:.2f}")
				except:
					st.metric("Average Confidence", "N/A")
				
				# Export results
				st.subheader("📤 Export Results")
				csv_data = df.to_csv(index=False)
				st.download_button(
					"Download CSV",
					csv_data,
					f"trading_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
					"text/csv"
				)
			else:
				st.info("No signal results to display")
	
	# Auto-refresh
	if st.session_state.get('run_signals', False) and auto_refresh:
		st.rerun()

elif page == "Reasoning Traces":
	st.subheader("🔍 Reasoning Trace Viewer")
	
	# Get reasoning traces
	try:
		response = httpx.get(f"{api_internal}/api/v1/reasoning/traces", timeout=5.0)
		if response.status_code == 200:
			traces_data = response.json()
			
			# Gracefully handle summary-only responses
			if traces_data.get("traces"):
				trace_ids = list(traces_data["traces"].keys())
				selected_trace_id = st.selectbox("Select Trace", trace_ids)
				if selected_trace_id:
					trace = traces_data["traces"][selected_trace_id]
					col1, col2 = st.columns([2,1])
					with col1:
						st.subheader("📋 Trace Details")
						st.write(trace)
					with col2:
						st.subheader("🎯 Final Decision")
						final_decision = trace.get("final_decision", {})
						if final_decision:
							action = final_decision.get("action", "unknown")
							confidence = final_decision.get("confidence", 0)
							color = "success" if action == "buy" else ("danger" if action == "sell" else "warn")
							st.markdown(f"<span class='badge {color}'>{action.upper()}</span>", unsafe_allow_html=True)
							st.metric("Confidence", f"{confidence:.2f}")
			else:
				# Show summary cards if only summary is available
				summary = traces_data.get("summary", {})
				c1, c2, c3 = st.columns(3)
				with c1:
					st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Total Traces</div><div class='kpi-value'>{summary.get('total_traces', 0)}</div></div>", unsafe_allow_html=True)
				with c2:
					st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Avg Duration (s)</div><div class='kpi-value'>{summary.get('avg_duration_seconds', 0):.3f}</div></div>", unsafe_allow_html=True)
				with c3:
					st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Decisions</div><div class='kpi-value'>{list(summary.get('decision_distribution', {}).keys())[:1] or ['-']}</div></div>", unsafe_allow_html=True)
		else:
			st.error(f"❌ Failed to fetch traces: {response.status_code}")
	except Exception as e:
		st.error(f"❌ Error fetching traces: {e}")

elif page == "System Health":
	st.subheader("🏥 System Health & Metrics")
	
	# Health check
	try:
		response = httpx.get(f"{api_internal}/health", timeout=5.0)
		if response.status_code == 200:
			health_data = response.json()
			components = health_data.get("components", {})
			grid = st.columns(3)
			for i, (comp, ok) in enumerate(components.items()):
				col = grid[i % 3]
				status_class = "success" if ok else "danger"
				col.markdown(f"<div class='kpi-card'><div class='kpi-title'>{comp.replace('_',' ').title()}</div><div class='kpi-value'><span class='badge {status_class}'>{'Healthy' if ok else 'Unhealthy'}</span></div></div>", unsafe_allow_html=True)
			
			st.markdown("<div class='section-title'>Prometheus (preview)</div>", unsafe_allow_html=True)
			metrics_response = httpx.get(f"{api_internal}/metrics", timeout=5.0)
			if metrics_response.status_code == 200:
				st.code(metrics_response.text[:1200] + ("..." if len(metrics_response.text) > 1200 else ""))
			else:
				st.error(f"Failed to fetch metrics: {metrics_response.status_code}")
		else:
			st.error(f"Health check failed: HTTP {response.status_code}")
	except Exception as e:
		st.error(f"❌ Health check failed: {e}")
