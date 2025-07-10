banner_placeholder = st.empty()

with banner_placeholder.container():
    banner(badges=None)  # Shows loading badges

cols = st.columns(2)

with cols[0]:
    ai_summary_text = activities()
    
    st.write(f"🔍 Debug: AI summary returned: {bool(ai_summary_text)}")
    
    if ai_summary_text:
        st.info("🎯 Generating personalized badges...")
        badges = get_ai_generated_badges(ai_summary_text)
        
        st.write(f"🔍 Debug: Badges received: {bool(badges)}")
        
        if badges:
            with banner_placeholder.container():
                banner(badges=badges)
            st.success("✅ Badges updated!")
        else:
            st.warning("⚠️ Badge generation failed - staying with loading badges")
    else:
        st.warning("⚠️ No AI summary text returned - cannot generate badges")

with cols[1]:
    opik_summary()
    em_summary()