banner_placeholder = st.empty()

with banner_placeholder.container():
    banner(badges=None)  # Shows loading badges

cols = st.columns(2)

with cols[0]:
    ai_summary_text = activities()
    
    if ai_summary_text:
        st.info("🎯 Generating personalized badges...")
        badges = get_ai_generated_badges(ai_summary_text)
        
        if badges:
            with banner_placeholder.container():
                banner(badges=badges)

with cols[1]:
    opik_summary()
    em_summary()