from object_manager import ObjectManager


def test_remove_nearest_point():
    manager = ObjectManager()

    manager.add_point(100, 100)
    manager.add_point(200, 200)
    manager.add_point(300, 300)

    removed = manager.remove_nearest_point(
        205,
        205,
        radius=20
    )

    assert removed == (200, 200)
    assert manager.current()["points"] == [
        (100, 100),
        (300, 300),
    ]


def test_remove_nearest_point_outside_radius():
    manager = ObjectManager()

    manager.add_point(100, 100)

    removed = manager.remove_nearest_point(
        300,
        300,
        radius=20
    )

    assert removed is None
    assert manager.current()["points"] == [
        (100, 100)
    ]


if __name__ == "__main__":
    test_remove_nearest_point()
    test_remove_nearest_point_outside_radius()
    print("ObjectManager removal tests passed.")
