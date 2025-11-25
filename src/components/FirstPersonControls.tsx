import { useRef, useEffect } from 'react';
import { useThree, useFrame } from '@react-three/fiber';
import * as THREE from 'three';

interface FirstPersonControlsProps {
  movementSpeed?: number;
  lookSpeed?: number;
}

export const FirstPersonControls = ({ movementSpeed = 2, lookSpeed = 0.002 }: FirstPersonControlsProps) => {
  const { camera, gl } = useThree();
  
  const moveForward = useRef(false);
  const moveBackward = useRef(false);
  const moveLeft = useRef(false);
  const moveRight = useRef(false);
  const moveUp = useRef(false);
  const moveDown = useRef(false);

  const velocity = useRef(new THREE.Vector3());
  const direction = useRef(new THREE.Vector3());
  
  // Accumulated rotation as floats to capture sub-pixel movements
  const euler = useRef(new THREE.Euler(0, 0, 0, 'YXZ'));
  const isLocked = useRef(false);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      switch (event.code) {
        case 'KeyW':
        case 'ArrowUp':
          moveForward.current = true;
          break;
        case 'KeyS':
        case 'ArrowDown':
          moveBackward.current = true;
          break;
        case 'KeyA':
        case 'ArrowLeft':
          moveLeft.current = true;
          break;
        case 'KeyD':
        case 'ArrowRight':
          moveRight.current = true;
          break;
        case 'Space':
          moveUp.current = true;
          break;
        case 'ShiftLeft':
          moveDown.current = true;
          break;
      }
    };

    const handleKeyUp = (event: KeyboardEvent) => {
      switch (event.code) {
        case 'KeyW':
        case 'ArrowUp':
          moveForward.current = false;
          break;
        case 'KeyS':
        case 'ArrowDown':
          moveBackward.current = false;
          break;
        case 'KeyA':
        case 'ArrowLeft':
          moveLeft.current = false;
          break;
        case 'KeyD':
        case 'ArrowRight':
          moveRight.current = false;
          break;
        case 'Space':
          moveUp.current = false;
          break;
        case 'ShiftLeft':
          moveDown.current = false;
          break;
      }
    };

    const handleMouseMove = (event: PointerEvent) => {
      if (!isLocked.current) return;

      const movementX = event.movementX || 0;
      const movementY = event.movementY || 0;

      // Log to verify we're getting sub-pixel precision
      console.log('Raw movement:', movementX, movementY, 'Type:', typeof movementX);

      // Accumulate rotations as floats - this ensures even tiny movements count
      euler.current.y -= movementX * lookSpeed;
      euler.current.x -= movementY * lookSpeed;

      // Clamp pitch to prevent flipping
      const PI_2 = Math.PI / 2;
      euler.current.x = Math.max(-PI_2, Math.min(PI_2, euler.current.x));
    };

    const handlePointerLockChange = () => {
      isLocked.current = document.pointerLockElement === gl.domElement;
    };

    const handleClick = () => {
      if (!isLocked.current) {
        gl.domElement.requestPointerLock();
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    document.addEventListener('keyup', handleKeyUp);
    // Use pointerrawupdate for high-precision mouse input instead of mousemove
    gl.domElement.addEventListener('pointerrawupdate', handleMouseMove);
    document.addEventListener('pointerlockchange', handlePointerLockChange);
    gl.domElement.addEventListener('click', handleClick);

    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      document.removeEventListener('keyup', handleKeyUp);
      gl.domElement.removeEventListener('pointerrawupdate', handleMouseMove);
      document.removeEventListener('pointerlockchange', handlePointerLockChange);
      gl.domElement.removeEventListener('click', handleClick);
    };
  }, [gl, lookSpeed]);

  useFrame((_, delta) => {
    if (!isLocked.current) return;

    // Apply accumulated rotation to camera
    camera.quaternion.setFromEuler(euler.current);

    // Handle movement
    velocity.current.x -= velocity.current.x * 10.0 * delta;
    velocity.current.y -= velocity.current.y * 10.0 * delta;
    velocity.current.z -= velocity.current.z * 10.0 * delta;

    direction.current.z = Number(moveForward.current) - Number(moveBackward.current);
    direction.current.x = Number(moveRight.current) - Number(moveLeft.current);
    direction.current.y = Number(moveUp.current) - Number(moveDown.current);
    direction.current.normalize();

    if (moveForward.current || moveBackward.current) velocity.current.z -= direction.current.z * movementSpeed * delta;
    if (moveLeft.current || moveRight.current) velocity.current.x -= direction.current.x * movementSpeed * delta;
    if (moveUp.current || moveDown.current) velocity.current.y += direction.current.y * movementSpeed * delta;

    // Get camera direction vectors
    const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion);
    const right = new THREE.Vector3(1, 0, 0).applyQuaternion(camera.quaternion);

    // Apply movement
    camera.position.addScaledVector(forward, -velocity.current.z * delta);
    camera.position.addScaledVector(right, -velocity.current.x * delta);
    camera.position.y += velocity.current.y * delta;
  });

  return null;
};
