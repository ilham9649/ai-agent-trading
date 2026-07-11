from trading_agent.models import EventKind
from trading_agent.state.audit_log import AuditLog


def test_chain_verifies(tmp_path):
    log = AuditLog(tmp_path / "a.sqlite")
    log.append(EventKind.HEARTBEAT, {"x": 1}, "c1")
    log.append(EventKind.FILL, {"y": 2}, "c1")
    assert log.verify()
    log.close()


def test_tamper_detected(tmp_path):
    log = AuditLog(tmp_path / "b.sqlite")
    log.append(EventKind.FILL, {"y": 2}, "c1")
    log.conn.execute("UPDATE events SET payload = ? WHERE id = 1", ('{"y": 99}',))
    log.conn.commit()
    assert not log.verify()
    log.close()


def test_hash_chaining(tmp_path):
    log = AuditLog(tmp_path / "c.sqlite")
    e1 = log.append(EventKind.HEARTBEAT, {"x": 1}, "c1")
    e2 = log.append(EventKind.HEARTBEAT, {"x": 2}, "c2")
    assert e1.prev_hash == AuditLog.GENESIS
    assert e2.prev_hash == e1.hash
    log.close()
