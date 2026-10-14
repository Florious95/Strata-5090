"""v0.1.44.1 cancellation/elastic vision combined with image refs/leases; CPU mocks only."""
import threading
from unittest import mock

from serve import server, test_vision_refs as fixtures


class Fusion(fixtures.Fixture):
    def test_bulk_cancel_keeps_lease_until_cleanup(self):
        cancel = threading.Event()
        encode = self.v.encode
        def first(source, **kwargs):
            row = encode(source, **kwargs)
            cancel.set()
            return row
        with self.v.lease() as lease, mock.patch.object(self.v, 'encode', side_effect=first):
            with self.assertRaises(server.ClientGone):
                self.v.encode_all([fixtures.png(), fixtures.png(1)], cancel=cancel, lease=lease)
            self.assertEqual(len(self.v.calls), 1)
            self.assertTrue(self.v.leased)
        self.assertFalse(self.v.leased)

    def test_elastic_lazy_init_also_initializes_reference_cache(self):
        cfg = {'exe': 'unused-encoder', 'mmproj': 'unused.gguf', 'model': 'unused-model.gguf',
               'elastic': True, 'elastic_mib': 77, 'elastic_idle_s': 10}
        with mock.patch.object(server.Vision, '_start') as start:
            vision = server.Vision(cfg)
            self.addCleanup(vision.shutdown)
            start.assert_not_called()
        self.assertTrue(vision.elastic)
        self.assertEqual((vision.elastic_mib, vision.elastic_idle_s), (77, 10))
        self.assertEqual((vision.cache, vision.raw_to_key, vision.key_to_raw), ({}, {}, {}))
        self.assertTrue(vision.cache_epoch)

    def test_cached_reference_with_cancel_does_not_wake_encoder_or_take_turn(self):
        data = fixtures.png()
        self.v.encode(data)
        req = {}
        with mock.patch.object(self.svc, '_turn', side_effect=AssertionError('cached ref took turn')), \
             mock.patch.object(self.svc, '_vision_up', side_effect=AssertionError('cached ref woke encoder')), \
             mock.patch.object(self.v, 'load', side_effect=AssertionError('cached ref decoded')):
            self.svc.prepare(self.messages([fixtures.reference(data)]), None, {}, max_new=8,
                             req=req, cancel=threading.Event())
        self.addCleanup(self.svc.drop_embeddings)
        self.assertEqual(req['_strata_image_refs'], [fixtures.reference(data)])
        self.assertEqual(len(self.v.calls), 1)
        self.assertFalse(self.v.leased)

    def test_tool_precheck_cancel_stops_next_picture_and_releases_lease(self):
        cancel = threading.Event()
        encode = self.v.encode
        def first(source, **kwargs):
            row = encode(source, **kwargs)
            cancel.set()
            return row
        with mock.patch.object(self.v, 'encode', side_effect=first), \
             mock.patch.object(self.svc, '_vision_up') as up:
            with self.assertRaises(server.ClientGone):
                self.svc.prepare(self.messages([fixtures.source(fixtures.png()),
                                                fixtures.source(fixtures.png(1))]),
                                 None, {}, max_new=8, cancel=cancel)
            up.assert_called_once()
        self.assertEqual(len(self.v.calls), 1)
        self.assertFalse(self.v.leased)
