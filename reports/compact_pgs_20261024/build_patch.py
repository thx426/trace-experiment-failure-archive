from pathlib import Path
import hashlib, json, compileall

ROOT=Path(__file__).resolve().parent
FROZEN=ROOT/'frozen/code'
PATCH=ROOT/'patch'

def edit(rel, replacements):
    text=(FROZEN/rel).read_text()
    for old,new in replacements:
        assert text.count(old)==1,(rel,old[:90],text.count(old))
        text=text.replace(old,new)
    out=PATCH/rel;out.parent.mkdir(parents=True,exist_ok=True);out.write_text(text)

edit('src/trace_opd/common/config.py',[
('    backend: str = "external"\n', '''    backend: str = "external"
    context_mode: str = "full"
    request_token_limit: int = 0
    context_target_tokens: int = 2500
    context_tokenizer_path: str = ""
    raw_page_tokens: int = 512
'''),
('                "backend": rcfg.backend,','''                "backend": rcfg.backend,
                "context_mode": rcfg.context_mode,
                "request_token_limit": rcfg.request_token_limit,
                "context_target_tokens": rcfg.context_target_tokens,
                "context_tokenizer_path": rcfg.context_tokenizer_path,
                "raw_page_tokens": rcfg.raw_page_tokens,
                "context_version": "compact3072_v1" if rcfg.context_mode == "bounded" else "full",
''')])

edit('src/trace_opd/common/llm.py',[
('    async def aclose(self) -> None:\n        await self._client.aclose()\n', '''    async def aclose(self) -> None:
        await self._client.aclose()

    async def count_prompt(self, messages, tools=None):
        # Same server model, native tools and template options as chat/completions.
        # This endpoint only tokenizes; it performs no model generation.
        payload = {"model": self.model, "messages": messages,
                   "add_generation_prompt": True, "add_special_tokens": False}
        if tools:
            payload["tools"] = tools
        for name in ("chat_template", "chat_template_kwargs", "add_generation_prompt",
                     "continue_final_message", "add_special_tokens"):
            if name in self._extra_body:
                payload[name] = self._extra_body[name]
        url = self.endpoint.rstrip("/").removesuffix("/v1") + "/tokenize"
        async with self._sem:
            response = await self._client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        count = data.get("count")
        if type(count) is not int or count <= 0 or count != len(data.get("tokens", [])):
            raise RuntimeError("invalid complete-template token count")
        return count
''')])

edit('src/trace_opd/train/agent_loop.py',[
('''        async def chat(messages, tools):
            return await client.chat(
                messages,
                tools=tools,
                max_tokens=rcfg.researcher_max_tokens,
                route_key=task_id,
            )

        return chat
''','''        async def count_tokens(messages, tools):
            from ..research.compact import contract_error
            try:
                return await client.count_prompt(messages, tools)
            except Exception as exc:
                contract_error(f"token counting failed: {type(exc).__name__}: {exc}")

        async def chat(messages, tools):
            bounded = rcfg.context_mode == "bounded"
            if bounded:
                from ..research.compact import contract_error
                if rcfg.request_token_limit != 3072 or rcfg.researcher_max_tokens != 4096:
                    contract_error("approved input/output limits changed")
                count = await count_tokens(messages, tools)
                if count > rcfg.request_token_limit:
                    contract_error(f"refusing {count}-token input before inference")
            result = await client.chat(
                messages,
                tools=tools,
                max_tokens=rcfg.researcher_max_tokens,
                route_key=task_id,
            )
            if bounded:
                if result.prompt_tokens != count or not 0 < result.prompt_tokens <= 3072:
                    contract_error(f"server usage mismatch: counted={count}, actual={result.prompt_tokens}")
                print(f"[compact3072] task={task_id} counted={count} actual={result.prompt_tokens}", flush=True)
            return result

        chat.count_tokens = count_tokens
        return chat
''')])

edit('src/trace_opd/research/session.py',[
('            while self._turns < self._max_turns and not state.finalized:',
 '            while self._turns < self._max_turns and not state.finalized and not getattr(self._compact, "rejected", False):'),
('        self._completion_tokens = 0\n','''        self._completion_tokens = 0
        self._compact = None
        if router.cfg.context_mode == "bounded":
            from .compact import CompactContext
            self._compact = CompactContext(router, max_turns)
            router._compact_context = self._compact
'''),
('                    result = await self._chat(self._messages, self._tools)','''                    if self._compact is None:
                        result = await self._chat(self._messages, self._tools)
                    else:
                        messages, tools, count = await self._compact.build(
                            self._turns, self._chat.count_tokens)
                        result = await self._chat(messages, tools)'''),
('                if result.text:\n','                if result.text or result.tool_calls:\n'),
('                    self._messages.append({"role": "user", "content": hint})\n','''                    self._messages.append({"role": "user", "content": hint})
                    if self._compact is not None:
                        self._compact.last_result = hint
                        self._compact.last_call = None
'''),
('                self._messages.append(self._proto.tool_result_message(first, tool_result))','''                self._messages.append(self._proto.tool_result_message(first, tool_result))
                if self._compact is not None:
                    self._compact.record(first, tool_result)''')])

edit('src/trace_opd/research/tools.py',[
('''        violations = self._check_doc(doc)
        if violations and not st.revision_requested:''','''        violations = self._check_doc(doc)
        if (getattr(self, "_compact_context", None) is not None
                and st.revision_requested and len(doc.split()) > self.cfg.doc_max_words):
            st.flags = violations
            st.no_finalize = True
            self._compact_context.rejected = True
            return "Document rejected: still exceeds hard word limit after one revision; no Skill emitted."
        if violations and not st.revision_requested:'''),
('''    def _fetch_raw(self, args: dict[str, Any]) -> str:
        st = self.state''','''    def _fetch_raw(self, args: dict[str, Any]) -> str:
        if getattr(self, "_compact_context", None) is not None:
            return self._compact_context.fetch_raw(args)
        st = self.state''')])

assert compileall.compile_dir(str(PATCH),quiet=1)
manifest={str(p.relative_to(PATCH)):hashlib.sha256(p.read_bytes()).hexdigest()
          for p in PATCH.rglob('*.py')}
(ROOT/'patch_manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2))
