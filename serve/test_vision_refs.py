"""Raw-SHA references and legacy image compatibility: fake encoder, no model/GPU."""
import base64
import copy
import hashlib
import io
import json
import struct
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from PIL import Image
from serve import server
from serve.frontend import openai_to_messages, anthropic_to_messages


def png(n=0):
    out = io.BytesIO()
    Image.new('RGB', (8, 8), (n % 256, 30, 40)).save(out, format='PNG')
    return out.getvalue()


def source(data):
    return 'data:image/png;base64,' + base64.b64encode(data).decode()


def reference(data):
    return 'strata-sha256://' + hashlib.sha256(data).hexdigest()


def make_vision(path):
    v = server.Vision.__new__(server.Vision)
    v.dir, v.lock, v.stopped = Path(path), threading.Lock(), False
    v._init_cache()
    calls = []
    class Pipe:
        last = ''
        def write(self, text):
            self.last = text
        def flush(self):
            if not self.last.startswith('ENC '):
                return
            _, img, out = self.last.split()
            data = (v.dir / img).read_bytes()
            calls.append(data)
            # Valid SVE shape; synthetic values, NOT real encoder output/quality evidence.
            value = hashlib.sha256(data).digest()[0] / 256.0
            (v.dir / out).write_bytes(struct.pack('<5i', 0x31455653, 3, 3, 1, 2560) +
                                     struct.pack('<f', value) * (3 * 2560))
        def readline(self):
            return 'OK 3 3 1 0\n'
    pipe = Pipe()
    v.proc = SimpleNamespace(stdin=pipe, stdout=pipe, poll=lambda: None, wait=lambda timeout=None: 0)
    v.calls = calls
    return v


class Cache(unittest.TestCase):
    def setUp(self):
        t = tempfile.TemporaryDirectory()
        self.addCleanup(t.cleanup)
        self.v = make_vision(t.name)
    def test_request_refs_are_isolated_and_do_not_decode_again(self):
        a, b = png(), png(1)
        with self.v.lease() as first:
            self.v.encode(source(a), lease=first)
            with self.v.lease() as second:
                self.v.encode(source(b), lease=second)
                self.assertEqual(second.refs, [reference(b)])
            with mock.patch.object(self.v, 'load', side_effect=AssertionError('reference must not load')):
                self.v.encode(reference(a), lease=first)
            self.assertEqual(first.refs, [reference(a), reference(a)])
        self.assertFalse(self.v.leased)

    def test_raw_base64_indexes_and_reference_are_identical(self):
        data = png()
        row = self.v.encode(source(data))
        raw, key = hashlib.sha256(data).hexdigest(), hashlib.sha256(data).hexdigest()[:32]
        self.assertEqual(self.v.raw_to_key[raw], key)
        self.assertEqual(self.v.key_to_raw[key], {raw})
        with mock.patch.object(self.v, 'load', side_effect=AssertionError('load')), \
             mock.patch.object(self.v, 'normalize', side_effect=AssertionError('normalize')):
            self.assertEqual(self.v.encode(reference(data)), row)
            self.assertEqual(self.v.encode('sha256:' + raw.upper()), row)
        self.assertEqual(len(self.v.calls), 1)
    def test_legacy_warm_raw_bytes_skip_normalization(self):
        data = png()
        row = self.v.encode(data)
        with mock.patch.object(self.v, 'normalize', side_effect=AssertionError('normalize')):
            self.assertEqual(self.v.encode(source(data)), row)
        self.assertEqual(len(self.v.calls), 1)
    def test_real_webp_normalization_is_only_once(self):
        out = io.BytesIO()
        Image.new('RGB', (8, 8), 'red').save(out, format='WEBP')
        data = out.getvalue()
        with mock.patch.object(self.v, 'normalize', wraps=self.v.normalize) as norm:
            a = self.v.encode(data)
            self.assertEqual(self.v.encode(source(data)), a)
            self.assertEqual(self.v.encode(reference(data)), a)
        self.assertEqual(norm.call_count, 1)
        self.assertEqual(len(self.v.calls), 1)
        self.assertNotEqual(hashlib.sha256(data).hexdigest()[:32], next(iter(self.v.cache)))
    def test_two_raw_aliases_to_one_normalized_entry(self):
        with mock.patch.object(self.v, 'normalize', return_value=png()):
            a, b = b'raw-a', b'raw-b'
            self.assertEqual(self.v.encode(a), self.v.encode(b))
        key = next(iter(self.v.cache))
        self.assertEqual(self.v.key_to_raw[key], {hashlib.sha256(a).hexdigest(), hashlib.sha256(b).hexdigest()})
        self.assertEqual(len(self.v.calls), 1)
    def test_aliases_are_bounded_without_deleting_shared_sve(self):
        with mock.patch.object(self.v, 'normalize', return_value=png()):
            for i in range(80):
                self.v.encode(f'raw-{i}'.encode())
        self.assertEqual(len(self.v.raw_to_key), 64)
        self.assertEqual(sum(map(len, self.v.key_to_raw.values())), 64)
        self.assertEqual(len(self.v.cache), 1)
        with self.assertRaises(server.ImageReferenceMissing):
            self.v.encode(reference(b'raw-0'))
    def test_normalized_eviction_removes_both_raw_indexes(self):
        old = png()
        row = self.v.encode(old)
        for i in range(1, 65):
            self.v.encode(png(i))
        raw = hashlib.sha256(old).hexdigest()
        self.assertNotIn(raw, self.v.raw_to_key)
        self.assertTrue(all(raw not in rs for rs in self.v.key_to_raw.values()))
        self.assertFalse(row[0].exists())
        with self.assertRaises(server.ImageReferenceMissing):
            self.v.encode(reference(old))
    def test_missing_file_is_a_reference_miss_and_legacy_recovers(self):
        data = png()
        row = self.v.encode(data)
        row[0].unlink()
        with self.assertRaises(server.ImageReferenceMissing):
            self.v.encode(reference(data))
        self.assertFalse(self.v.raw_to_key)
        self.assertFalse(self.v.key_to_raw)
        self.assertEqual(self.v.encode(data), row)
        self.assertTrue(row[0].is_file())
        self.assertEqual(len(self.v.calls), 2)
    def test_reference_does_not_need_a_running_encoder(self):
        data = png()
        row = self.v.encode(data)
        self.v.unload()
        self.assertEqual(self.v.encode(reference(data)), row)
        self.assertEqual(len(self.v.calls), 1)
    def test_malformed_reference_is_not_read_as_a_file(self):
        for value in ('strata-sha256://abc', 'sha256:' + 'g' * 64, reference(png()) + '/x'):
            with self.subTest(value=value), mock.patch.object(self.v, 'load', side_effect=AssertionError('file read')):
                with self.assertRaises(ValueError):
                    self.v.encode(value)
    def test_request_lease_survives_concurrent_eviction_and_releases(self):
        old = png()
        row = self.v.encode(old)
        with self.v.lease() as held:
            self.v.encode(reference(old), lease=held)
            errors = []
            def worker():
                try:
                    for i in range(1, 70):
                        self.v.encode(png(i))
                except Exception as e:
                    errors.append(e)
            t = threading.Thread(target=worker)
            t.start(); t.join(10)
            self.assertFalse(t.is_alive())
            self.assertFalse(errors)
            self.assertTrue(row[0].exists())
        self.assertFalse(self.v.leased)
        self.assertLessEqual(len(self.v.cache), 64)
    def test_failed_encode_does_not_publish_raw_alias(self):
        self.v.proc.stdin.write = mock.Mock(side_effect=BrokenPipeError('gone'))
        with self.assertRaises(BrokenPipeError):
            self.v.encode(png())
        self.assertFalse(self.v.raw_to_key)
        self.assertFalse(self.v.key_to_raw)
        self.assertFalse(list(self.v.dir.glob('*.img')))


class Fixture(unittest.TestCase):
    def setUp(self):
        t = tempfile.TemporaryDirectory()
        self.addCleanup(t.cleanup)
        self.v = make_vision(t.name)
        tok = server.ByteTokenizer()
        self.svc = server.Service(server.MockEngine(tok, '</think>\n\nok', max_context=262144), tok,
                                  server.ChatTemplate(server.ROOT / 'serve/chat_template.jinja'), vision=self.v)
    def messages(self, sources, role='tool'):
        return [{'role': 'user', 'content': 'review'}, {'role': role, 'content':
                 [{'type': 'text', 'text': 'history'}] + [{'type': 'image', 'source': s} for s in sources]}]
    def prepare(self, messages):
        ids, _, _ = self.svc.prepare(copy.deepcopy(messages), None, {}, max_new=8)
        p = self.svc.embeddings.path
        sve = p.read_bytes() if p else b''
        self.svc.drop_embeddings()
        return ids, sve


class Preparation(Fixture):
    def test_17_tool_images_no_double_load_normalize_or_enc(self):
        data = [png(i) for i in range(17)]
        legacy = self.messages([source(d) for d in data])
        with mock.patch.object(self.v, 'load', wraps=self.v.load) as load, \
             mock.patch.object(self.v, 'normalize', wraps=self.v.normalize) as norm:
            a = self.prepare(legacy)
            self.assertEqual((load.call_count, norm.call_count, len(self.v.calls)), (17, 17, 17))
            load.reset_mock(); norm.reset_mock()
            self.assertEqual(self.prepare(legacy), a)
            self.assertEqual((load.call_count, norm.call_count, len(self.v.calls)), (17, 0, 17))
            load.reset_mock(); norm.reset_mock()
            self.assertEqual(self.prepare(self.messages([reference(d) for d in data])), a)
            self.assertEqual((load.call_count, norm.call_count, len(self.v.calls)), (0, 0, 17))
    def test_mixed_raw_ref_and_new_image_preserves_ids_and_bytes(self):
        data = [png(i) for i in range(3)]
        self.v.encode(data[0]); self.v.encode(data[1])
        mixed = self.prepare(self.messages([source(data[0]), reference(data[1]), source(data[2])]))
        self.assertEqual(mixed, self.prepare(self.messages([source(d) for d in data])))
        self.assertEqual(len(self.v.calls), 3)
    def test_duplicate_image_occurrences_keep_duplicate_records(self):
        data = png()
        a = self.prepare(self.messages([source(data), source(data)]))
        b = self.prepare(self.messages([reference(data), reference(data)]))
        self.assertEqual(a, b)
        self.assertEqual(len(self.v.calls), 1)
        self.assertEqual(len(a[1]), 2 * (20 + 3 * 2560 * 4))
    def test_missing_tool_references_are_strict_and_all_reported(self):
        refs = [reference(png(1)), reference(png(2))]
        with self.assertRaises(server.ImageReferenceMissing) as caught:
            self.prepare(self.messages(refs))
        self.assertEqual(caught.exception.hashes, [r.split('://')[1] for r in refs])
        self.assertFalse(self.v.calls)
        self.assertFalse(self.v.leased)
    def test_malformed_tool_reference_not_omitted(self):
        with self.assertRaises(ValueError):
            self.prepare(self.messages(['strata-sha256://bad']))
    def test_legacy_bad_tool_still_note_and_user_still_error(self):
        with mock.patch.object(self.v, 'load', side_effect=ValueError('bad image')):
            ids, sve = self.prepare(self.messages(['bad.png']))
            self.assertIn('[image omitted: bad image]', self.svc.tok.decode(ids))
            self.assertFalse(sve)
            with self.assertRaises(ValueError):
                self.prepare(self.messages(['bad.png'], role='user'))
    def test_http_tool_download_and_precheck_only_once(self):
        with mock.patch.object(server.Vision, 'download', return_value=png()) as download:
            self.prepare(self.messages(['https://example.test/a.png']))
        self.assertEqual(download.call_count, 1)
        self.assertEqual(len(self.v.calls), 1)
    def test_file_legacy_and_reference_same(self):
        data = png()
        f = self.v.dir / 'legacy.png'; f.write_bytes(data)
        a = self.prepare(self.messages([str(f)], role='user'))
        self.assertEqual(self.prepare(self.messages([reference(data)], role='user')), a)
    def test_reference_hits_do_not_wait_for_fifo_or_encoder(self):
        data = png(); self.v.encode(data)
        self.svc.fifo.acquire()
        results = []
        t = threading.Thread(target=lambda: results.append(self.prepare(self.messages([reference(data)]))))
        try:
            t.start(); t.join(5)
            self.assertFalse(t.is_alive(), 'cached refs must not acquire the legacy encoder gate')
            self.assertEqual(len(results), 1)
        finally:
            self.svc.fifo.release(); t.join(5)
    def test_failed_prepare_releases_cache_lease(self):
        data = png(); self.v.encode(data)
        with self.assertRaises(ValueError):
            self.svc.prepare(self.messages([reference(data)]), None, {}, max_new=999999)
        self.assertFalse(self.v.leased)
        self.assertIsNone(self.svc.embeddings.path)
    def test_more_than64_images_survive_precheck_then_trim(self):
        ids, sve = self.prepare(self.messages([source(png(i)) for i in range(70)]))
        self.assertTrue(ids)
        self.assertEqual(len(sve), 70 * (20 + 3 * 2560 * 4))
        self.assertFalse(self.v.leased)
        self.assertEqual(len(self.v.cache), 64)
    def test_chat_and_anthropic_reference_prefix_contract(self):
        data = png(); self.v.encode(data)
        for converter, reqs in (
            (openai_to_messages, [{'messages': [{'role': 'user', 'content': [{'type': 'image_url',
                'image_url': {'url': s}}]}]} for s in (source(data), reference(data))]),
            (anthropic_to_messages, [{'messages': [{'role': 'user', 'content': [{'type': 'image',
                'source': {'type': 'url', 'url': s}}]}]} for s in (source(data), reference(data))])):
            results = []
            for req in reqs:
                m, tools, kw = converter(req)
                ids, _, _ = self.svc.prepare(m, tools, kw, max_new=8)
                results.append((ids, self.svc.embeddings.path.read_bytes()))
                self.svc.drop_embeddings()
            self.assertEqual(*results)


class API(Fixture):
    def setUp(self):
        super().setUp()
        self.svc.api_key = 'fixture-test-key'
        self.httpd = server.serve(self.svc, port=0)
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)
        self.url = 'http://127.0.0.1:' + str(self.httpd.server_address[1])
    def request(self, path, body=None, authorized=True, origin=None):
        headers = {'Content-Type': 'application/json'}
        if authorized:
            headers['Authorization'] = 'Bearer fixture-test-key'
        if origin:
            headers['Origin'] = origin
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode() if body is not None else None,
                                     headers=headers)
        try:
            response = urllib.request.urlopen(req, timeout=10)
        except urllib.error.HTTPError as e:
            response = e
        with response:
            self.last_headers = response.headers
            return response.status, json.load(response)
    def test_query_is_authenticated_and_does_not_encode(self):
        data = png(); self.v.encode(data)
        path = '/v1/vision/cache?sha256=' + hashlib.sha256(data).hexdigest()
        self.assertEqual(self.request(path, authorized=False)[0], 401)
        status, body = self.request(path)
        self.assertEqual(status, 200)
        self.assertEqual(body['namespace'], self.v.cache_epoch)
        self.assertTrue(body['images'][hashlib.sha256(data).hexdigest()]['cached'])
        self.assertEqual(len(self.v.calls), 1)
    def test_props_advertises_raw_hash_not_normalized_digest(self):
        status, body = self.request('/props')
        self.assertEqual(status, 200)
        self.assertEqual(body['image_references']['hash'], 'sha256-raw-bytes')
    def test_missing_tool_ref_400_no_generation(self):
        ref = reference(png())
        payload = {'model': self.svc.model, 'max_tokens': 8, 'messages': [
            {'role': 'tool', 'content': [{'type': 'image_url', 'image_url': {'url': ref}}]}]}
        status, body = self.request('/v1/chat/completions', payload)
        self.assertEqual(status, 400)
        self.assertEqual(body['error']['code'], 'image_cache_miss')
        self.assertEqual(body['error']['missing_sha256'], [ref.split('://')[1]])
        self.assertEqual(self.svc.engine.last_prompt, [])
        self.assertEqual(self.svc.engine.turns, 0)
    def test_legacy_then_ref_http200_same_prompt(self):
        data = png()
        for src in (source(data), reference(data)):
            status, body = self.request('/v1/chat/completions', {'model': self.svc.model, 'max_tokens': 8,
                'messages': [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': src}}]}]})
            self.assertEqual(status, 200, body)
            self.assertEqual(json.loads(self.last_headers['X-Strata-Image-Refs']), [reference(data)])
            ids = self.svc.engine.last_prompt
            if src.startswith('data:'):
                old = ids
            else:
                self.assertEqual(ids, old)
        self.assertEqual(len(self.v.calls), 1)
    def test_query_validation_and_unknown_digest(self):
        self.assertEqual(self.request('/v1/vision/cache?sha256=bad')[0], 400)
        status, body = self.request('/v1/vision/cache?sha256=' + 'f' * 64)
        self.assertEqual(status, 200)
        self.assertFalse(body['images']['f' * 64]['cached'])
    def test_foreign_origin_reference_does_not_open_file_paths(self):
        self.svc.cors_origins = ['*']
        data = png(); self.v.encode(data)
        for src, expected in ((reference(data), 200), (str(self.v.dir / 'secret.png'), 400)):
            status, _ = self.request('/v1/chat/completions', {'model': self.svc.model, 'max_tokens': 8,
                'messages': [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': src}}]}]},
                origin='https://foreign.test')
            self.assertEqual(status, expected)
            if expected == 200:
                self.assertEqual(self.last_headers['Access-Control-Expose-Headers'], 'X-Strata-Image-Refs')


if __name__ == '__main__':
    unittest.main()
