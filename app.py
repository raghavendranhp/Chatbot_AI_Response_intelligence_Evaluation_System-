import streamlit as st
import time
from src.orchestrator import CruiseOrchestrator
from evaluation.engine import EvaluationEngine


st.set_page_config(
    page_title="SeShat AI - Egypt Cruises",
    page_icon="🚢",
    layout="wide"
)


@st.cache_resource
def get_system():
    """
    Initializes the Chatbot and the Evaluator once.
    """
    #Load Chatbot
    orchestrator = CruiseOrchestrator()
    
    #Load Evaluator
    
    evaluator = EvaluationEngine(
        llm_callable=orchestrator.generate_response, 
        ground_truth_path=r"D:\Projects\chatbot_seshat\data\ground_truth\Egypt_Cruise_GroundTruth_Dataset.csv",
        log_dir="logs"
    )
    return orchestrator, evaluator

try:
    orchestrator, evaluator = get_system()
except Exception as e:
    st.error(f"System failed to load: {e}")
    st.stop()


st.sidebar.title("🛠️ Control Panel")

st.sidebar.subheader("AI Intelligence Layer")
enable_eval = st.sidebar.toggle("Enable Live Evaluation", value=True, help="Runs the Quality Assurance engine on every response.")

st.sidebar.divider()
st.sidebar.info(
    "**System Status:**\n"
    "🟢 Chatbot: Online\n"
    "🟢 Evaluator: Online\n"
    "🟢 Guardrails: Active"
)

st.title("🚢 SeShat AI: Egypt Cruise Assistant")
st.caption("Powered by RAG + Self-Correcting Intelligence Layer")

#Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

#Display History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        #If this message has an evaluation report attached, display it
        if "eval_report" in msg:
            report = msg["eval_report"]
            score = report['overall_score']
            
            #Color-code the score
            if score >= 0.8:
                score_color = "green"
                status_icon = "✅"
            elif score >= 0.6:
                score_color = "orange"
                status_icon = "⚠️"
            else:
                score_color = "red"
                status_icon = "❌"

            with st.expander(f"{status_icon} AI Confidence Score: :{score_color}[{score}/1.0]"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Relevance", report['scores']['components'].get('relevance_score', 0))
                c2.metric("Consistency", report['scores']['components'].get('consistency_score', 0))
                c3.metric("Rule Check", report['scores']['rule_adherence'])
                
                if report['flags']:
                    st.error(f"**Risk Flags:** {report['flags']}")
                else:
                    st.success("No hallucinations or errors detected.")


if prompt := st.chat_input("Ask about pricing, itineraries, or specific cruises..."):
    #Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    #Process & Respond
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("Thinking & Checking Ground Truth..."):
            try:
                #Generate Response (Returns dict with 'answer' and 'context')
                response_obj = orchestrator.process_query(prompt)
                answer_text = response_obj["answer"]
                context_text = response_obj["context"]
                
                #Run Evaluation (If enabled)
                eval_report = None
                if enable_eval:
                    eval_report = evaluator.evaluate(prompt, answer_text, context_text)
                
                #Display Answer
                message_placeholder.markdown(answer_text)
                
                #Display Live Score Card
                if eval_report:
                    score = eval_report['overall_score']
                    
                    if score >= 0.8:
                        score_color = "green"
                        status_msg = "High Confidence"
                    elif score >= 0.6:
                        score_color = "orange"
                        status_msg = "Review Needed"
                    else:
                        score_color = "red"
                        status_msg = "Low Quality / Risk"

                    st.divider()
                    st.caption(f" **AI Guardrails:** {status_msg} | Score: :{score_color}[{score}]")
                    
                    if eval_report['flags']:
                        st.warning(f"Flagged Issues: {eval_report['flags']}")

                #Save to History
                msg_data = {"role": "assistant", "content": answer_text}
                if eval_report:
                    msg_data["eval_report"] = eval_report
                st.session_state.messages.append(msg_data)

            except Exception as e:
                st.error(f"An error occurred: {e}")