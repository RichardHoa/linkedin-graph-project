import time
import json
import os
from flask import Flask, request, jsonify, send_from_directory
from neo4j import GraphDatabase
from groq import Groq

app = Flask(__name__)

# --- CONFIGURATION ---
neo4j_driver = GraphDatabase.driver(
    "neo4j+s://55115469.databases.neo4j.io", 
    auth=("neo4j", "tTe604uH7jxZ-ZFyk7Jr6F19rqGqVxgBpZpi6Nfs8HA")
)

# Initialize Groq Client
groq_client = Groq(api_key="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") 
MODEL_NAME = "llama-3.3-70b-versatile"

with open('graph-structure.json', 'r') as f:
    schema_data = json.load(f)

def format_schema_for_llm(data):
    schema_summary = "GRAPH DATABASE SCHEMA:\n"
    for item in data:
        label = item['label']
        details = item['details']
        if details['type'] == 'node':
            props = list(details.get('properties', {}).keys())
            rels = list(details.get('relationships', {}).keys())
            schema_summary += f"- Node [{label}]: Properties {props} | Relationships {rels}\n"
        else:
            schema_summary += f"- Relationship [{label}]: Properties {list(details.get('properties', {}).keys())}\n"
    return schema_summary

def execute_cypher(query):
    start_time = time.time()
    try:
        with neo4j_driver.session() as session:
            result = session.run(query)
            data = [record.data() for record in result]
            return data, None, (time.time() - start_time)
    except Exception as e:
        return None, str(e), (time.time() - start_time)

def summarize_results(results, user_question):
    summary_prompt = (
        f"Summarize these {len(results)} raw database records into concise, factual insights "
        f"specifically answering: '{user_question}'. Do not lose specific counts or unique entities."
    )
    response = groq_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": summary_prompt}, {"role": "user", "content": str(results)}],
        temperature=0
    )
    return response.choices[0].message.content

GRAPH_CONTEXT = format_schema_for_llm(schema_data)

# GENERALIZED SYSTEM PROMPT
SYSTEM_PROMPT = f"""
Act as a Senior Graph Data Scientist specializing in multi-hop Retrieval-Augmented Generation. 
You are responsible for extracting insights from a graph database regardless of the domain (career paths, education, or organizations).

SCHEMA CONTEXT:
{GRAPH_CONTEXT}

CORE OPERATIONAL RULES:
1. MANDATORY CLARIFICATION: On the first interaction, you MUST only ask a clarifying question to define the scope of the user's request. DO NOT generate Cypher or analysis yet.
2. NO HALLUCINATION: You are FORBIDDEN from generating statistics (e.g., "30% of people...") unless those exact numbers are returned by a Cypher query.
3. FUZZY MATCHING: Use `toLower(n.name) CONTAINS toLower('search_term')` for all string lookups.
4. SEARCH RELAXATION: If a specific role or entity search returns 0 results, broaden your next search to a higher-level category or related attribute.
5. REPORTING: Always state the sample size (n=X) based on the query results.

CYPHER GUIDELINES:
- Map user entities to nodes like :Professional, :JobTitle, :Experience, or :University based on schema.
- Transitions: (e1:Experience)-[:FOLLOWED_BY]->(e2:Experience).
- Always verify relationship directions in the schema before writing Cypher.
"""

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/ask', methods=['POST'])
def handle_query():
    overall_start = time.time()
    data = request.json
    user_question = data.get('question', '')
    history_data = data.get('history', []) 
    further_question = data.get('furtherQuestion', False)
    
    messages = []
    for entry in history_data:
        role = "assistant" if entry['role'] == "model" else entry['role']
        messages.append({"role": role, "content": entry['text']})
    messages.append({"role": "user", "content": user_question})



    if not further_question and len(history_data) == 0:
        response = groq_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages
        )
        return jsonify({"question": response.choices[0].message.content, "currentHop": 0})

    query_budget = 0
    while query_budget < 3:
        hop_id = query_budget + 1

        if further_question:
            hop_context = (
                f"This is a follow-up (HOP {hop_id}/3). "
                "IMPORTANT: If the data needed to answer is already in the conversation history, "
                "DO NOT generate a new Cypher query. Just type 'ANALYZE'. "
                "Only query Neo4j if you need new, specific data."
            )
        else:
            hop_context = (
                f"This is HOP {hop_id}/3. If you do not have specific data yet, "
                "you MUST generate a ```cypher query. If you have enough data, type 'ANALYZE'."
            )

        
        response = groq_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages + [{"role": "user", "content": hop_context}],
            temperature=0.1
        )
        
        decision = response.choices[0].message.content
        if "```cypher" not in decision:
            break

        try:
            cypher_query = decision.split("```cypher")[1].split("```")[0].strip()
            db_results, error, _ = execute_cypher(cypher_query)

            if error:
                messages.append({"role": "user", "content": f"Cypher error: {error}"})
                continue

            query_budget += 1
            if db_results:
                summary = summarize_results(db_results, user_question)
                feedback = f"HOP {hop_id} INSIGHTS (n={len(db_results)}): {summary}"
            else:
                feedback = f"HOP {hop_id} RESULT: 0 records found."
                
            messages.append({"role": "assistant", "content": decision})
            messages.append({"role": "user", "content": feedback})
        except Exception:
            break

    messages = [
        m for m in messages 
        if not (m['role'] == 'assistant' and '```cypher' in m['content']) 
        and not (m['role'] == 'user' and m['content'].startswith("Cypher error:"))
    ]

    # 3. FINAL SYNTHESIS
    all_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages + [{"role": "user", "content": "Synthesize final answer."}]
    print(all_messages)
    final_response = groq_client.chat.completions.create(
        model=MODEL_NAME,
        messages=all_messages,
        temperature=0.5
    )
    
    return jsonify({
        "analysis": final_response.choices[0].message.content,
        "isFinal": True
    })

if __name__ == '__main__':
    app.run(port=5000, debug=True)